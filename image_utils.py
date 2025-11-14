# image_utils.py

import base64
import io
import time
import random
from enum import Enum
import openai
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, AutoPipelineForText2Image
import re
import string
import json 

# --- Image & Text Helper Functions --- 

MODEL_SPECS = {
    "SD15": {
        "model_id": "sd-legacy/stable-diffusion-v1-5",
        "height": 512, "width": 512,
        "num_inference_steps": 30,
        "guidance_scale": 7.5,
    },
    "SDXL_Turbo": {
        "model_id": "stabilityai/sdxl-turbo",
        "height": 1024, "width": 1024,
        "num_inference_steps": 1,
        "guidance_scale": 0.0,
    },
    "FLUX_1_dev": {
        "model_id": "black-forest-labs/FLUX.1-dev",
        "height": 1024, "width": 1024,
        "num_inference_steps": 30,
        "guidance_scale": 3.5,
    },
    "SD35_medium": {
        "model_id": "stabilityai/stable-diffusion-3.5-medium",
        "height": 1024, "width": 1024,
        "num_inference_steps": 30,
        "guidance_scale": 4.5,
    },
}

def encode_image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def load_diffusion_model(model_path, device):
    print(f"Loading image generation model: {model_path}...")
    try:
        if "sdxl-turbo" in model_path:
            spec = MODEL_SPECS["SDXL_Turbo"]
        elif "FLUX.1-dev" in model_path:
            spec = MODEL_SPECS["FLUX_1_dev"]
        elif "stable-diffusion-3.5-medium" in model_path:
            spec = MODEL_SPECS["SD35_medium"]
        else:
            spec = MODEL_SPECS["SD15"]
        pipeline = AutoPipelineForText2Image.from_pretrained(spec["model_id"], torch_dtype=torch.bfloat16).to(device)
        print("Image generation model loaded successfully!")
        return pipeline, spec
    except Exception as e:
        raise RuntimeError(f"Failed to load diffusion model: {e}")

def load_target_image(image_path: str):
    print(f"Loading target image: {image_path}")
    try:
        return Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        raise FileNotFoundError(f"Target image not found at {image_path}.")

def get_caption_from_image(image: Image.Image, vlm_model_name: str) -> str:
    print(f"Generating initial prompt using {vlm_model_name}...")
    instruction = "Analyze the image and provide a detailed, high-quality prompt... Your response must be only the prompt itself."
    base64_image = encode_image_to_base64(image)
    
    response = openai.chat.completions.create(
        model=vlm_model_name,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": instruction},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]}],
        max_tokens=150
    )
    return response.choices[0].message.content.strip()

def generate_image_from_prompt(prompt: str, pipeline, spec, args):
    if pipeline is None: return Image.new('RGB', (512, 512), color='black')
    try:
        generator = torch.Generator(device=pipeline.device).manual_seed(args.seed)
        return pipeline(prompt,
                height=spec["height"],
                width=spec["width"],
                num_inference_steps=spec["num_inference_steps"],
                guidance_scale=spec["guidance_scale"],
                generator=generator).images[0]
    except Exception as e:
        print(f"Error during image generation: {e}")
        return Image.new('RGB', (512, 512), color='grey')

def calculate_clip_similarity(image_a, image_b, clip_model, clip_preprocess, device):
    if clip_model is None: return 0.0
    processed_a = clip_preprocess(image_a).unsqueeze(0).to(device)
    processed_b = clip_preprocess(image_b).unsqueeze(0).to(device)
    with torch.no_grad():
        feat_a = clip_model.encode_image(processed_a)
        feat_b = clip_model.encode_image(processed_b)
        feat_a /= feat_a.norm(dim=-1, keepdim=True)
        feat_b /= feat_b.norm(dim=-1, keepdim=True)
        return (feat_a @ feat_b.T).item()

def execute(mutate_results, target_image, diffusion_pipeline, spec, clip_model, clip_preprocess, device, args):
    prompt_template = mutate_results[0] if isinstance(mutate_results, list) else mutate_results
    generated_image = generate_image_from_prompt(prompt_template, diffusion_pipeline, spec, args)
    similarity_score = calculate_clip_similarity(generated_image, target_image, clip_model, clip_preprocess, device)
    return similarity_score, generated_image, prompt_template

# --- MCTS and Fuzzing State Management ---

class mutator(Enum):
    GEN_DESC_STYLE     = 0  
    MODIFY_STYLE       = 1  
    MODIFY_DESC        = 2  
    PARAPHRASE_BASE    = 3  
    ENRICH_BASE_INLINE = 4  
    FIX_GRAMMAR        = 5  


class prompt_node:
    def __init__(
        self, 
        text,                
        relation,             
        style, 
        base_prompt=None,     
        parent=None, 
        generation=0, 
        mutation=None, 
        index=0, 
        response=0.0
    ):
        self.text = text
        self.relation = relation
        self.style = style
        self.base_prompt = base_prompt if base_prompt is not None else text  
        self.parent = parent
        self.generation = parent.generation + 1 if parent and hasattr(parent, 'generation') else generation
        self.mutation = mutation
        self.index = index
        self.response = response
        self.children = []
        self.visited_num = 0
        self.mcts_reward = 0.0

class fuzzing_status:
    def __init__(self, initial_nodes, max_query=-1):
        self.max_query = max_query
        self.query = len(initial_nodes)
        self.seed_queue = initial_nodes
        self.max_seed_pool_size = 5
        self.seed_selection_strategy = self.seed_selection_MCTS
        self.pointer = 0

    def stop_condition(self):
        return self.max_query != -1 and self.query >= self.max_query

    def seed_selection_MCTS(self):
        if not self.seed_queue: return None
        total_visits = sum(node.visited_num for node in self.seed_queue) + 1
        best_node = max(self.seed_queue, key=lambda n: 
            (n.mcts_reward / (n.visited_num + 1e-5)) + 
            1.5 * np.sqrt(np.log(total_visits) / (n.visited_num + 1e-5))
        )
        self.pointer = self.seed_queue.index(best_node)
        return best_node

    def update_with_node(self, new_node: prompt_node):
        self.query += 1
        if len(self.seed_queue) < self.max_seed_pool_size:
            self.seed_queue.append(new_node)
        else:
            worst_node = min(self.seed_queue, key=lambda n: n.response)
            if new_node.response > worst_node.response:
                self.seed_queue.remove(worst_node)
                self.seed_queue.append(new_node)
        
        if self.pointer < len(self.seed_queue):
            node_to_update = self.seed_queue[self.pointer]
            node_to_update.visited_num += 1
            node_to_update.mcts_reward += new_node.response

    def get_best_seed(self):
        if not self.seed_queue: 
            return prompt_node("N/A", "N/A", "N/A", response=-1)
        return max(self.seed_queue, key=lambda node: node.response)

# --- Mutation Operator Logic ---
def mutate_operator(base_prompt: str, mutation_type: mutator, parent_node=None, base_only: bool=False):
    if base_only:
        base_only_set = [mutator.ENRICH_BASE_INLINE, mutator.FIX_GRAMMAR, mutator.PARAPHRASE_BASE]
        if mutation_type not in base_only_set:
            mutation_type = random.choice(base_only_set)

    system_prompt = (
        "you are a prompt mutator for text-to-image diffusion models.\n"
        "given a base prompt and an input image, you must return EXACTLY ONE SINGLE-LINE JSON object.\n"
        "- lowercase english only; no markdown; no code fences; no trailing commas; no extra text.\n"
        "- never insert line breaks inside values.\n"
        "- if the base prompt conflicts with the image, trust the image.\n"
        "- be concrete and visual; do not invent invisible objects.\n"
        "- field-specific rules:\n"
        "  description: 15-35 words, write as a compact comma-separated tag string (not full sentences). include: subject and key attributes or pose, setting/location, composition/shot/angle, lighting, overall color tendency, one brief quality token (e.g., highly detailed or sharp focus), optional material/texture cue, artist, plus up to 2 simple negatives (e.g., no watermark, no text). use only visible facts.\n"

        "  style: <= 12 words, a short comma-separated tag string of medium/movement/lens/quality only (e.g., digital painting, photorealistic, vector art, isometric, 35mm lens, 85mm lens, film grain, clean render). do not include scene facts or lighting.\n"
        "  base_prompt: <= 15 words, preserve the original meaning, clearer phrasing, avoid style or lighting tokens.\n"
        "examples:\n"
        " input(base): 'two samurai duel in a bamboo forest'\n"
        " output(desc+style): {\"description\":\"bamboo grove, two samurai facing between tall stalks, medium shot, eye-level, dappled sunlight, green tones, foreground leaves, no text\",\"style\":\"ink illustration, ukiyo-e inspired, paper texture\"}\n"
        " input(base): 'astronaut and robot on mars at dawn'\n"
        " output(modify-desc): {\"description\":\"red dunes, astronaut left of small robot, wide shot, low angle, soft dawn light, cool shadows, distant mountains, no watermark\"}\n"
        " input(base): 'portrait of an old musician in neon city'\n"
        " output(modify-style): {\"style\":\"photorealistic, 85mm lens, cinematic still\"}\n"
        " input(base): 'a child reading a book under a tree'\n"
        " output(paraphrase-base): {\"base_prompt\":\"a child reading beneath a tree\"}\n"
    )

    if mutation_type == mutator.ENRICH_BASE_INLINE:
        user_prompt = (
            "task: ENRICH the base prompt by inserting concise, image-grounded modifiers WITHOUT changing its syntactic skeleton or word order.\n"
            f"base prompt: '{base_prompt}'\n"
            "you will receive the IMAGE together with this message. use ONLY details that are VISIBLE in the image.\n"
            "output schema (single line json): {\"base_prompt\":\"...\"}\n"
            "constraints:\n"
            "- preserve the original tokens as an ordered subsequence: do not delete, replace, or reorder existing words; no synonym substitution.\n"
            "- keep the subject–verb–object–prepositional structure intact.\n"
            "- INSERT 2–6 words total, placed immediately AFTER the nouns/verbs they modify (adjectives/appositives for nouns; short adverbs/adjuncts for verbs).\n"
            "- allowed insertions: count/quantity (one/two), object attributes (color, size, material), pose/state, simple spatial cues relative to BACKGROUND (e.g., near the fence, in front of the gate), and other concrete scene facts visible in the image.\n"
            "- forbid insertions about style/lighting/lens/artist or abstract aesthetics (these belong to style).\n"
            "- if a detail is uncertain or not visible, DO NOT add it; trust the image over the text.\n"
            "- lowercase only; no quotes; no extra commentary.\n"
            "examples:\n"
            " base: 'a child reading a book under a tree'\n"
            " enriched: {'base_prompt':'a small child quietly reading a worn book under a shady tree'}\n"
            " base: 'a dog runs across a field'\n"
            " enriched: {'base_prompt':'a brown dog runs swiftly across a grassy field'}\n"
            "return ONLY the json line."
        )

    elif mutation_type == mutator.FIX_GRAMMAR:
        user_prompt = (
            "task: CLEANUP the base prompt by fixing grammar/spelling, removing duplicated words/phrases, and correcting spacing/punctuation ONLY.\n"
            f"base prompt: '{base_prompt}'\n"
            "output schema (single line json): {\"base_prompt\":\"...\"}\n"
            "constraints:\n"
            "- do NOT add any new content or modifiers; do NOT introduce synonyms; do NOT reorder clauses.\n"
            "- preserve the original subject–verb–object–prepositional order and overall sentence structure.\n"
            "- only remove repeated tokens/phrases, fix typos, collapse multiple spaces, and standardize minimal punctuation.\n"
            "- keep length approximately unchanged (within ±2 words of the original).\n"
            "- lowercase only; no quotes; no extra commentary.\n"
            "return ONLY the json line."
        )

    elif mutation_type == mutator.GEN_DESC_STYLE:
        user_prompt = (
            "task: generate a NEW description and a NEW style from the base prompt and the image.\n"
            "base prompt: '{base_prompt}'\n"
            'output schema (single line json): {"description":"...","style":"..."}\n'
            "constraints:\n"
            "- description must be a compact comma-separated tag string (15–35 words), not full sentences.\n"
            "- include, in order when possible: subject with salient attributes or pose, setting/location, composition/shot/angle, lighting, overall color tendency, one brief quality token (e.g., highly detailed or sharp focus), optional material/texture cue, artist, optional negatives (max 2: no watermark, no text).\n"
            "- include one explicit spatial or depth cue relating subject(s) to background (e.g., foreground, background, in front of distant hills).\n"
            "- use only concrete, visible details; avoid story terms and abstractions.\n"
            "- style contains only medium/movement/lens/quality tokens (≤12 words); no scene facts and no lighting; no artist names.\n"
            "return ONLY the json line.\n"
        )

    elif mutation_type == mutator.MODIFY_STYLE:
        if not parent_node:
            return system_prompt, (
                "task: generate a NEW description and a NEW style from the base prompt and the image.\n"
                f"base prompt: '{base_prompt}'\n"
                "output schema: {\"description\":\"...\",\"style\":\"...\"}\n"
                "return ONLY the json line."
            )
        user_prompt = (
            "task: CHANGE STYLE ONLY while preserving entities and relations in the current description.\n"
            f"current style: {getattr(parent_node, 'style', '')}\n"
            f"current description: {getattr(parent_node, 'relation', getattr(parent_node, 'description', ''))}\n"
            f"base prompt: '{base_prompt}'\n"
            'output schema (single line json): {"style":"..."}\n'
            "constraints:\n"
            "- produce a concise comma-separated tag string (≤12 words) consisting only of medium/movement/lens/quality tokens (e.g., digital painting, photorealistic, vector art, isometric, 35mm lens, 85mm lens, film grain, clean render).\n"
            "- do NOT include scene facts or lighting\n"
            "- keep the aesthetic consistent with the base and description; improve clarity or fidelity rather than altering content.\n"
            "return ONLY the json line.\n"
        )

    elif mutation_type == mutator.MODIFY_DESC:
        if not parent_node:
            return system_prompt, (
                "task: generate a NEW description and a NEW style from the base prompt and the image.\n"
                f"base prompt: '{base_prompt}'\n"
                "output schema: {\"description\":\"...\",\"style\":\"...\"}\n"
                "return ONLY the json line."
            )
        user_prompt = (
            "task: CHANGE DESCRIPTION ONLY while preserving the subject meaning and current style.\n"
            f"current description: {getattr(parent_node, 'relation', getattr(parent_node, 'description', ''))}\n"
            f"current style: {getattr(parent_node, 'style', '')}\n"
            f"base prompt: '{base_prompt}'\n"
            'output schema (single line json): {"description":"..."}\n'
            "constraints:\n"
            "- produce a compact comma-separated tag string (15–35 words), not full sentences.\n"
            "- include, in order when possible: subject and attributes/pose, setting/location, composition/shot/angle, lighting, overall color tendency, one quality token (e.g., highly detailed or sharp focus), optional material/texture, artist, optional negatives (max 2).\n"
            "- include one explicit spatial or depth cue (e.g., foreground, background, in front of, near).\n"
            "- keep to visible, concrete details only; avoid story/abstract words.\n"
            "- do not change the style semantics.\n"
            "return ONLY the json line.\n"

        )

    elif mutation_type == mutator.PARAPHRASE_BASE:
        user_prompt = (
            "task: PARAPHRASE the base prompt WITHOUT changing its meaning, and enforce the structure:\n"
            "WHO/WHAT + is doing + WHERE.\n"
            f"base prompt: '{base_prompt}'\n"
            "output schema (single line json): {\"base_prompt\":\"...\"}\n"
            "constraints:\n"
            "- <= 15 words total.\n"
            "- structure must be strictly: <who/what> + 'is' + <present participle verb phrase> + <where-phrase>.\n"
            "- examples of valid skeletons:\n"
            "  'a child is reading under a tree'\n"
            "  'two samurai are dueling in a bamboo forest'\n"
            "  'a red car is driving along a rainy street'\n"
            "- who/what: a concrete subject noun phrase (singular/plural ok: 'a child' / 'two samurai').\n"
            "- is doing: present progressive ('is/are' + V-ing), one concise action.\n"
            "- where: a concrete location or scene prepositional phrase (e.g., 'under a tree', 'in a market').\n"
            "- no style/lighting/lens/artist tokens.\n"
            "- preserve entities and their core relations from the base prompt; trust the image if there is a conflict.\n"
            "- lowercase only; no quotes; no extra commentary.\n"
            "return ONLY the json line."
        )

    else:
        user_prompt = (
            "task: generate a NEW description and a NEW style from the base prompt and the image.\n"
            f"base prompt: '{base_prompt}'\n"
            "output schema: {\"description\":\"...\",\"style\":\"...\"}\n"
            "return ONLY the json line."
        )

    return system_prompt, user_prompt


def mutate_single(base_prompt: str, mutation_type: mutator, args, target_image: Image.Image, parent_node=None, PROCESSOR=None, VL_MODEL=None) -> tuple[str, str, str]:
    system_prompt, user_prompt = mutate_operator(base_prompt, mutation_type, parent_node, base_only=args.base_only)
    
    mutated_text = ""
    if str(args.image_model_path).startswith("gpt-"):
        try:
            base64_image = encode_image_to_base64(target_image)
            image_data_url = f"data:image/jpeg;base64,{base64_image}"
            messages = [{"role": "user", "content": [
                {"type": "text", "text": f"{system_prompt}\n\n{user_prompt}"},
                {"type": "image_url", "image_url": {"url": image_data_url}}
            ]}]
            response = openai.chat.completions.create(
                model=args.image_model_path,
                messages=messages,
                temperature=1.0, 
                n=1,
                max_tokens=200
            )
            mutated_text = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM Vision API request failed: {e}")
    else:
        mutated_text = local_vl_request(system_prompt, user_prompt, args.target_image_path, PROCESSOR, VL_MODEL, args)

    parsed = parse_llm_response(mutated_text)  # dict: {description, style, base_prompt}
    description = parsed.get("description", "")  # relation == description
    style = parsed.get("style", "")
    new_base = parsed.get("base_prompt", "")

    if mutation_type == mutator.MODIFY_STYLE and parent_node:
        if not style:
            style = parent_node.style  
        if description == "":
            description = getattr(parent_node, 'relation', getattr(parent_node, 'description', ""))

    if mutation_type == mutator.MODIFY_DESC and parent_node:
        if not description:
            description = getattr(parent_node, 'relation', getattr(parent_node, 'description', ""))
        if style == "":
            style = parent_node.style

    if mutation_type == mutator.GEN_DESC_STYLE:
        if not description and parent_node:
            description = getattr(parent_node, 'relation', getattr(parent_node, 'description', ""))
        if not style and parent_node:
            style = parent_node.style

    if mutation_type == mutator.PARAPHRASE_BASE:
        if new_base:
            base_prompt = clean_text(new_base) or base_prompt
        if parent_node:
            if not description:
                description = getattr(parent_node, 'relation', getattr(parent_node, 'description', ""))
            if not style:
                style = parent_node.style

    if mutation_type in {mutator.ENRICH_BASE_INLINE, mutator.FIX_GRAMMAR, mutator.PARAPHRASE_BASE}:
        if new_base:
            base_prompt = clean_text(new_base) or base_prompt
        if parent_node:
            if not description:
                description = getattr(parent_node, 'relation', getattr(parent_node, 'description', ""))
            if not style:
                style = parent_node.style

    description = clean_text(description)
    style = clean_text(style)
    parts = [p for p in [base_prompt, description, style] if p]
    full_prompt = ", ".join(parts).strip(" ,")

    return full_prompt, base_prompt, description, style


def parse_llm_response(llm_output: str) -> dict:
    def _clean_val(s: str) -> str:
        if not isinstance(s, str):
            return ""
        s = s.strip()
        s = s.replace("\n", " ").replace("\r", " ")
        s = re.sub(r"\s{2,}", " ", s)
        return s.lower().strip(' "\'')

    def _extract_json_block(text: str) -> str | None:
        if not text:
            return None
        t = text.strip()
        t = re.sub(r"^```(?:json|JSON)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
        m = re.search(r"\{.*\}", t, flags=re.DOTALL)
        return m.group(0) if m else None

    def _try_load_json(s: str) -> dict | None:
        if not s:
            return None
        s = re.sub(r",\s*}", "}", s)
        s = re.sub(r",\s*]", "]", s)
        if "'" in s and '"' not in s:
            s = s.replace("'", '"')
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
        return None

    desc = sty = bp = ""
    if not isinstance(llm_output, str) or not llm_output.strip():
        return {"description": desc, "style": sty, "base_prompt": bp}

    raw = llm_output.strip()
    json_blk = _extract_json_block(raw)
    if json_blk:
        obj = _try_load_json(json_blk)
        if obj is not None:
            desc = _clean_val(obj.get("description", ""))
            sty  = _clean_val(obj.get("style", ""))
            bp   = _clean_val(obj.get("base_prompt", ""))
            return {"description": desc, "style": sty, "base_prompt": bp}

    m_desc = re.search(r"(?i)\bdescription\s*:\s*(.+)", raw)
    m_style = re.search(r"(?i)\bstyle\s*:\s*(.+)", raw)
    m_base = re.search(r"(?i)\bbase[_\s-]?prompt\s*:\s*(.+)", raw)

    def _slice_until_next_key(s: str) -> str:
        if not s:
            return ""
        s = re.split(r"(?i)\b(?:description|style|base[_\s-]?prompt)\s*:", s)[0]
        s = s.strip().rstrip(",;")
        return s

    if m_desc:
        desc = _clean_val(_slice_until_next_key(m_desc.group(1)))
    if m_style:
        sty = _clean_val(_slice_until_next_key(m_style.group(1)))
    if m_base:
        bp = _clean_val(_slice_until_next_key(m_base.group(1)))

    return {"description": desc, "style": sty, "base_prompt": bp}


def clean_text(text: str) -> str:
    if not isinstance(text, str): return ""
    text = text.lower()
    text = text.replace('\n', ' ').replace('\r', '').replace('[', '').replace(']', '')
    return text.strip()

def local_vl_request(system_prompt: str, user_prompt: str, image_path: str, PROCESSOR, VL_MODEL, args):
    """
    Sends a request to a local Vision-Language Model using its processor.
    """
    try:
        image = Image.open(image_path).convert("RGB")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image",},
                {"type": "text", "text": user_prompt},
            ]},
        ]
        text = PROCESSOR.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = PROCESSOR(text=text, images=image, return_tensors="pt").to(VL_MODEL.device)
        with torch.no_grad():
            output = VL_MODEL.generate(
                **inputs,
                do_sample=True, top_p=1.0, top_k=50, temperature=1.0,
                repetition_penalty=1.05, max_new_tokens=200
            )
        response_text = PROCESSOR.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        cleaned_response = response_text.split("assistant")[-1].strip()
        return cleaned_response
    except Exception as e:
        print(f"Error during local VL model request: {e}")
        return "Local model request failed."
