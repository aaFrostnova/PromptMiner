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
import json  # <-- 新增

# --- Image & Text Helper Functions --- (这部分无变化)

def encode_image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def load_diffusion_model(model_path, device):
    print(f"Loading image generation model: {model_path}...")
    try:
        pipeline_class = AutoPipelineForText2Image if "sdxl" in model_path else StableDiffusionPipeline
        pipeline = pipeline_class.from_pretrained(model_path, torch_dtype=torch.float16)
        pipeline.to(device)
        print("Image generation model loaded successfully!")
        return pipeline
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

def generate_image_from_prompt(prompt: str, pipeline, args):
    if pipeline is None: return Image.new('RGB', (512, 512), color='black')
    try:
        generator = torch.Generator(device=pipeline.device).manual_seed(args.seed)
        if "sdxl" in args.target_model_path:
            return pipeline(prompt=prompt, num_inference_steps=1, guidance_scale=0, generator=generator).images[0]
        else:
            return pipeline(prompt=prompt, num_inference_steps=50, generator=generator).images[0]
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

def execute(mutate_results, target_image, diffusion_pipeline, clip_model, clip_preprocess, device, args):
    prompt_template = mutate_results[0] if isinstance(mutate_results, list) else mutate_results
    generated_image = generate_image_from_prompt(prompt_template, diffusion_pipeline, args)
    similarity_score = calculate_clip_similarity(generated_image, target_image, clip_model, clip_preprocess, device)
    return similarity_score, generated_image, prompt_template

# --- MCTS and Fuzzing State Management ---

class mutator(Enum):
    GEN_DESC_STYLE     = 0  # 生成新的 description + style
    MODIFY_STYLE       = 1  # 只改 style
    MODIFY_DESC        = 2  # 只改 description
    PARAPHRASE_BASE    = 3  # 保持原意重写 base prompt
    ENRICH_BASE_INLINE = 4  # ★ 在不改句法骨架与词序的前提下，基于图像内联补细节
    FIX_GRAMMAR        = 5  # ★ 仅修正语法/拼写/重复词与空格标点；不添加信息

# === prompt_node 数据结构（保留原字段名以最小改动；relation == description） ===
class prompt_node:
    def __init__(
        self, 
        text,                 # full prompt (base + desc + style)
        relation,             # description
        style, 
        base_prompt=None,     # <--- 新增
        parent=None, 
        generation=0, 
        mutation=None, 
        index=0, 
        response=0.0
    ):
        self.text = text
        self.relation = relation
        self.style = style
        self.base_prompt = base_prompt if base_prompt is not None else text  # fallback 避免None
        self.parent = parent
        self.generation = parent.generation + 1 if parent and hasattr(parent, 'generation') else generation
        self.mutation = mutation
        self.index = index
        self.response = response
        self.children = []
        self.visited_num = 0
        self.mcts_reward = 0.0

# fuzzing_status 类无需修改，因为它操作的是通用的 prompt_node 对象
class fuzzing_status:
    """
    更新后的 fuzzing_status 类，初始化方法现在直接接收节点列表。
    """
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

# ========================
# mutate_operator (替换版)
# ========================
def mutate_operator(base_prompt: str, mutation_type: mutator, parent_node=None, base_only: bool=False):
    """
    base_prompt 相关:
      - ENRICH_BASE_INLINE: 在不改变句法骨架与词序的前提下插入可见事实
      - FIX_GRAMMAR      : 仅修正语法/去重/空格标点，不添加信息
      - PARAPHRASE_BASE  : 保持原意，但重写表述（非 base_only 时可用）

    modifiers:
      - GEN_DESC_STYLE: 生成新的 description 与 style
      - MODIFY_STYLE  : 仅重写 style（媒介/流派/镜头/质感）
      - MODIFY_DESC   : 仅重写 description（场景/关系/构图/光线/色彩/负面）

    输出均为 单行 JSON；英文字母小写；严格长度约束；如图像与文本冲突，以图像为准。
    """

    if base_only:
        base_only_set = [mutator.ENRICH_BASE_INLINE, mutator.FIX_GRAMMAR, mutator.PARAPHRASE_BASE]
        if mutation_type not in base_only_set:
            mutation_type = random.choice(base_only_set)

    # ===== 通用 system prompt（Qwen/vLLM 多模态友好，强化“关系”） =====
    system_prompt = (
        "you are a prompt mutator for text-to-image diffusion models.\n"
        "given a base prompt and an input image, you must return EXACTLY ONE SINGLE-LINE JSON object.\n"
        "- lowercase english only; no markdown; no code fences; no trailing commas; no extra text.\n"
        "- never insert line breaks inside values.\n"
        "- if the base prompt conflicts with the image, trust the image.\n"
        "- be concrete and visual; do not invent invisible objects.\n"
        "- field-specific rules:\n"
        "  description: 15-35 words, visible details only. include: \n"
        "    • setting/environment (scene/location),\n"
        "    • explicit relations: use 1–3 relation tokens when >=2 subjects (choose from: left of, right of, in front of, behind, above, below, between, next to, near, around, inside, atop, facing, overlapping, surrounding; plus depth cues: foreground, midground, background),\n"
        "    • composition/shot/angle (e.g., close-up, medium shot, wide shot, low angle, high angle, eye-level, rule of thirds, centered, symmetrical),\n"
        "    • lighting (e.g., soft light, rim light, backlight, golden hour, overcast),\n"
        "    • color tendency (e.g., warm tones, cool tones, high contrast),\n"
        "    • optional material/texture or foreground/background cue, and optional 0–2 negatives (e.g., no watermark, no text).\n"
        "    if there is only one subject, still describe its relation to the background or depth (e.g., in front of mountains, foreground with distant city).\n"
        "    avoid artist names.\n"
        "  style: <= 12 words, only medium/movement/lens/quality tokens (e.g., watercolor, art nouveau, 85mm lens, film grain). no scene facts.\n"
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

    # ===== base-only 变异 =====
    if mutation_type == mutator.ENRICH_BASE_INLINE:
        # 基于图像内联补细节：不改变句法骨架与词序，只插入可见事实
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
        # 仅修正语法/拼写/重复词与空格标点；不添加信息
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

    # ===== 其余（非 base-only）任务 =====
    elif mutation_type == mutator.GEN_DESC_STYLE:
        # 生成新的 description + style（要求显式关系与层次）
        user_prompt = (
            "task: generate a NEW description and a NEW style from the base prompt and the image.\n"
            f"base prompt: '{base_prompt}'\n"
            "output schema (single line json): {\"description\":\"...\",\"style\":\"...\"}\n"
            "constraints:\n"
            "- description must include: setting + explicit relations (use 1–3 relation tokens if >=2 subjects, or relate the single subject to background/depth) + composition/shot/angle + lighting + color tendency.\n"
            "- use relation tokens from: left of, right of, in front of, behind, above, below, between, next to, near, around, inside, atop, facing, overlapping, surrounding; and depth terms: foreground, midground, background.\n"
            "- style contains only medium/movement/lens/quality tokens.\n"
            "return ONLY the json line."
        )

    elif mutation_type == mutator.MODIFY_STYLE:
        # 仅修改 style（保持描述与关系不变）
        if not parent_node:
            # 无父节点时退化为生成式
            return system_prompt, (
                "task: generate a NEW description and a NEW style from the base prompt and the image.\n"
                f"base prompt: '{base_prompt}'\n"
                "output schema: {\"description\":\"...\",\"style\":\"...\"}\n"
                "return ONLY the json line."
            )
        user_prompt = (
            "task: CHANGE STYLE ONLY while preserving the meaning and relations in the current description.\n"
            f"current style: {getattr(parent_node, 'style', '')}\n"
            f"current description: {getattr(parent_node, 'relation', getattr(parent_node, 'description', ''))}\n"
            f"base prompt: '{base_prompt}'\n"
            "output schema (single line json): {\"style\":\"...\"}\n"
            "constraints:\n"
            "- style <= 12 words; medium/movement/lens/quality only; no scene facts; no artist names.\n"
            "- do not change entities or their relations implied by the current description.\n"
            "return ONLY the json line."
        )

    elif mutation_type == mutator.MODIFY_DESC:
        # 仅修改 description（强调加入/修正“关系”）
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
            "output schema (single line json): {\"description\":\"...\"}\n"
            "constraints:\n"
            "- 15-35 words; include setting/environment AND explicit relations (use 1–3 relation tokens if >=2 subjects, or relate the single subject to background/depth), plus composition/shot/angle, lighting, color tendency.\n"
            "- allowed relation tokens: left of, right of, in front of, behind, above, below, between, next to, near, around, inside, atop, facing, overlapping, surrounding; and depth terms: foreground, midground, background.\n"
            "- keep style semantics unchanged; do not add artist names.\n"
            "return ONLY the json line."
        )

    elif mutation_type == mutator.PARAPHRASE_BASE:
        # 保持原意不变，重写 base prompt —— 强制 Who/What + is doing + where 结构
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
        # 兜底：退化为生成式
        user_prompt = (
            "task: generate a NEW description and a NEW style from the base prompt and the image.\n"
            f"base prompt: '{base_prompt}'\n"
            "output schema: {\"description\":\"...\",\"style\":\"...\"}\n"
            "return ONLY the json line."
        )

    return system_prompt, user_prompt


# === 更新 mutate_single 来处理新的输入输出(JSON优先) ===
def mutate_single(base_prompt: str, mutation_type: mutator, args, target_image: Image.Image, parent_node=None, PROCESSOR=None, VL_MODEL=None) -> tuple[str, str, str]:
    """
    执行单次多模态变异，返回 (full_prompt, relation, style)
    其中 relation == description（为了最小改动复用原接口）
    """

    # 在 operator 中也传入 base_only，用于必要时强制回退 mutation 类型
    system_prompt, user_prompt = mutate_operator(base_prompt, mutation_type, parent_node, base_only=args.base_only)
    
    mutated_text = ""
    # 调用 VLM 的部分（兼容 gpt-* 或本地 vLLM，如 Qwen）
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

    # --- 解析新返回 ---
    parsed = parse_llm_response(mutated_text)  # dict: {description, style, base_prompt}
    description = parsed.get("description", "")  # relation == description
    style = parsed.get("style", "")
    new_base = parsed.get("base_prompt", "")

    # --- 根据任务类型进行继承/兜底 ---
    if mutation_type == mutator.MODIFY_STYLE and parent_node:
        if not style:
            style = parent_node.style  # 解析失败则继承旧的
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

    # ★ base-only 情况下（仅修改 base_prompt）：统一处理 ENRICH_BASE_INLINE / FIX_GRAMMAR
    if mutation_type in {mutator.ENRICH_BASE_INLINE, mutator.FIX_GRAMMAR, mutator.PARAPHRASE_BASE}:
        if new_base:
            base_prompt = clean_text(new_base) or base_prompt
        # 不改变描述与风格；若为空则尽量继承父节点，便于拼装 full_prompt
        if parent_node:
            if not description:
                description = getattr(parent_node, 'relation', getattr(parent_node, 'description', ""))
            if not style:
                style = parent_node.style

    # --- 清理文本 ---
    description = clean_text(description)
    style = clean_text(style)


    # 仅当存在 style 才加前缀；base_only 模式不会强行生成 style
    if style and not style.startswith("in the style of"):
        style = f"in the style of {style}"

    # --- 构建最终的 prompt： [base_prompt], [description], [style] ---
    parts = [p for p in [base_prompt, description, style] if p]
    full_prompt = ", ".join(parts).strip(" ,")

    # 返回：full_prompt, relation(==description), style
    return full_prompt, base_prompt, description, style


# === 新的解析器：JSON 优先，键值行回退 ===
def parse_llm_response(llm_output: str) -> dict:
    """
    统一返回:
      {
        "description": str,  # 若没给则为 ""
        "style": str,        # 若没给则为 ""
        "base_prompt": str   # 若没给则为 ""
      }
    解析优先级：JSON > 行内键值对 > 兜底空串
    """
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
        # 去除可能的代码块包裹
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

    # 1) 优先 JSON
    json_blk = _extract_json_block(raw)
    if json_blk:
        obj = _try_load_json(json_blk)
        if obj is not None:
            desc = _clean_val(obj.get("description", ""))
            sty  = _clean_val(obj.get("style", ""))
            bp   = _clean_val(obj.get("base_prompt", ""))
            return {"description": desc, "style": sty, "base_prompt": bp}

    # 2) 回退：解析 "key: value" 行
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

# --- 辅助函数 (无变化) ---
def clean_text(text: str) -> str:
    if not isinstance(text, str): return ""
    text = text.lower()
    # 标点符号在这里可能有用 (e.g., '8k'), 所以只移除换行符
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
