# utils_image.py

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
# --- Image & Text Helper Functions ---

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
    GENERATE_NEW = 0
    REFINE_DETAIL = 1
    CHANGE_POSITION = 2

class prompt_node:
    def __init__(self, text, detail, position, parent=None, generation=0, mutation=None, index=0, response=0.0):
        self.text = text
        self.detail = detail
        self.position = position
        self.parent = parent
        self.generation = parent.generation + 1 if parent and hasattr(parent, 'generation') else generation
        self.mutation = mutation
        self.index = index
        self.response = response 
        self.children = []
        self.visited_num = 0
        self.mcts_reward = 0.0

class fuzzing_status:
    """
    更新后的 fuzzing_status 类，初始化方法现在直接接收节点列表。
    """
    def __init__(self, initial_nodes, max_query=-1):
        self.max_query = max_query
        # 查询次数应从初始节点的数量开始
        self.query = len(initial_nodes)
        # 直接使用传入的节点列表作为种子队列
        self.seed_queue = initial_nodes
        self.max_seed_pool_size = 15 # 您可以根据需要调整种子池大小
        self.seed_selection_strategy = self.seed_selection_MCTS
        self.pointer = 0

    def stop_condition(self):
        return self.max_query != -1 and self.query >= self.max_query

    def seed_selection_MCTS(self):
        if not self.seed_queue: return None
        total_visits = sum(node.visited_num for node in self.seed_queue) + 1
        
        # UCT 公式： 探索 (exploration) + 利用 (exploitation)
        # 增加了探索权重，鼓励尝试更多可能性
        best_node = max(self.seed_queue, key=lambda n: 
            (n.mcts_reward / (n.visited_num + 1e-5)) + 
            1.5 * np.sqrt(np.log(total_visits) / (n.visited_num + 1e-5))
        )
        
        self.pointer = self.seed_queue.index(best_node)
        return best_node

    def update_with_node(self, new_node: prompt_node):
        self.query += 1
        
        # 精英池管理策略
        if len(self.seed_queue) < self.max_seed_pool_size:
            self.seed_queue.append(new_node)
        else:
            # 如果新节点的分数高于池中最差的节点，则替换
            worst_node = min(self.seed_queue, key=lambda n: n.response)
            if new_node.response > worst_node.response:
                self.seed_queue.remove(worst_node)
                self.seed_queue.append(new_node)
        
        # MCTS 反向传播更新
        # 确保指针有效
        if self.pointer < len(self.seed_queue):
            node_to_update = self.seed_queue[self.pointer]
            node_to_update.visited_num += 1
            node_to_update.mcts_reward += new_node.response

    def get_best_seed(self):
        if not self.seed_queue: 
            return prompt_node("N/A", "N/A", "N/A", response=-1)
        return max(self.seed_queue, key=lambda node: node.response)

# --- Mutation Operator Logic ---

# --- 文本突变函数 ---
# 在 utils_image.py 中，替换掉旧的 mutate_operator 函数
def mutate_operator(base_prompt: str, mutation_type: mutator, parent_node=None):
    # 系统指令保持不变
    system_prompt = (
        "You are an expert prompt engineer for text-to-image models. "
        "Your task is to analyze the provided image and complete a base prompt by providing text for 'detail' and 'position' that accurately describe the image. "
        "Your response MUST strictly follow the format 'detail: [text]\\nposition: [text]' and contain nothing else."
    )

    # --- 任务1: (基于图片) 生成全新的 detail 和 position ---
    if mutation_type == mutator.GENERATE_NEW:
        user_prompt = f"""Given the base prompt:
            '{base_prompt}'

            Analyze the attached image and generate a compelling 'detail' and 'position' to complete the prompt. The generated text must describe the visual content of the image.

            # Constraints:
            # - The 'detail' MUST be under 15 words.
            # - The 'position' MUST be under 10 words."""

    # --- 任务2: (基于图片) 优化已有的 detail ---
    elif mutation_type == mutator.REFINE_DETAIL:
        if not parent_node:
            return mutate_operator(base_prompt, mutator.GENERATE_NEW)
        
        user_prompt = f"""Given the base prompt:
            '{base_prompt}'

            And the current completion:
            detail: {parent_node.detail}
            position: {parent_node.position}

            Now, analyze the attached image and refine ONLY the 'detail' part to better describe the visual content. Keep the 'position' exactly the same.

            # Constraint:
            # - The new 'detail' MUST be under 15 words."""

    # --- 任务3: (基于图片) 创造新的 position ---
    elif mutation_type == mutator.CHANGE_POSITION:
        if not parent_node:
            return mutate_operator(base_prompt, mutator.GENERATE_NEW)
            
        user_prompt = f"""Given the base prompt:
            '{base_prompt}'

            And the current completion:
            detail: {parent_node.detail}
            position: {parent_node.position}

            Now, analyze the attached image and invent a new 'position' or camera angle that better suits the image content. Keep the 'detail' exactly the same.

            # Constraint:
            # - The new 'position' MUST be under 10 words."""
    else:
        return mutate_operator(base_prompt, mutator.GENERATE_NEW)

    return system_prompt, user_prompt
        

def mutate_single(base_prompt: str, mutation_type: mutator, args, target_image: Image.Image, parent_node=None, PROCESSOR=None, VL_MODEL=None) -> tuple[str, str, str]:
    """
    执行单次多模态变异操作，将图片和文本指令一起发送给 LLM。
    """
    # 1. 生成文本指令
    system_prompt, user_prompt = mutate_operator(base_prompt, mutation_type, parent_node)
    
    mutated_text = ""
    if args.image_model_path.startswith("gpt-"):
        try:
            # 2. 将图片编码为 Base64
            base64_image = encode_image_to_base64(target_image)
            image_data_url = f"data:image/jpeg;base64,{base64_image}"

            # 3. 构建多模态的 messages payload
            messages = [
                # Vision 模型通常将 system prompt 的内容合并到 user message 中
                {
                    "role": "user",
                    "content": [
                        # 首先是文本指令
                        {"type": "text", "text": f"{system_prompt}\n\n{user_prompt}"},
                        # 然后是图片
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data_url}
                        }
                    ]
                }
            ]

            # 4. 调用 OpenAI Vision API
            response = openai.chat.completions.create(
                model=args.image_model_path,  # 确保这里使用的是 gpt-4o 或其他 vision 模型
                messages=messages,
                temperature=1.0, 
                n=1,
                max_tokens=50 
            )
            mutated_text = response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"LLM Vision API request failed: {e}")
    
    else:
        mutated_text = local_vl_request(system_prompt, user_prompt, args.target_image_path, PROCESSOR, VL_MODEL, args)

    # 5. 解析并清理结果 (这部分逻辑不变)
    detail, position = parse_llm_response(mutated_text)
    detail = clean_text(detail)
    position = clean_text(position)
    
    # 6. 拼接最终的 prompt (这部分逻辑不变)
    connector = " " if not base_prompt.endswith(" ") and not detail.startswith(" ") else ""
    if detail == "" and parent_node:
        full_prompt = f"{base_prompt}{connector}{parent_node.detail}, {position}"
    elif position == "" and parent_node:
        full_prompt = f"{base_prompt}{connector}{detail}, {parent_node.position}"
    else:
        full_prompt = f"{base_prompt}{connector}{detail}, {position}"
    
    return full_prompt, detail, position


def parse_llm_response(llm_output: str) -> tuple[str, str]:
    """
    一个健壮的解析器，用于从 LLM 的原始输出中提取 'detail' 和 'position'。

    Args:
        llm_output (str): 从 LLM API 返回的原始字符串。

    Returns:
        tuple[str, str]: 一个包含 (detail, position) 的元组。
                         如果找不到某个字段，会使用默认值。
    """
    # 默认值，用于处理解析失败或字段缺失的情况
    default_detail = ""
    default_position = ""
    print(f"[LLM Output] {llm_output}")
    try:
        # 使用正则表达式进行不区分大小写的匹配
        # re.DOTALL 标志让 '.' 可以匹配换行符，这对于多行的 detail/position 至关重要
        detail_match = re.search(r"detail:\s*(.*)", llm_output, re.IGNORECASE | re.DOTALL)
        position_match = re.search(r"position:\s*(.*)", llm_output, re.IGNORECASE | re.DOTALL)

        # 提取匹配到的内容，并移除首尾的空白字符
        # 如果没有匹配到，则使用默认值
        detail = detail_match.group(1).strip() if detail_match else default_detail
        position = position_match.group(1).strip() if position_match else default_position
        
        # 进一步处理：有时 position 可能会被 detail 的多行内容捕获，我们需要切分它
        if detail_match and position_match and detail.find("position:") != -1:
             detail = detail.split("position:")[0].strip()

        return detail, position

    except Exception as e:
        print(f"[Parser Error] Failed to parse LLM output: {e}")
        return default_detail, default_position



# 将这个新函数添加到你的 utils_image.py 文件中，可以放在其他辅助函数旁边
def clean_text(text: str) -> str:
    """
    清理文本字符串，移除所有标点符号并将所有字符转换为小写。

    Args:
        text (str): 输入的字符串。

    Returns:
        str: 清理和格式化后的字符串。
    """
    if not isinstance(text, str):
        return ""
        
    # 步骤 1: 将所有字符转换为小写
    text = text.lower()
    
    # 步骤 2: 移除所有标点符号
    # string.punctuation 包含了一系列常见的标点符号
    # str.translate() 是一个高效移除多个字符的方法
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    
    return text

def local_vl_request(system_prompt: str, user_prompt: str, image_path: str, PROCESSOR, VL_MODEL, args):
    """
    Sends a request to a local Vision-Language Model using its processor.
    """
    try:
        image = Image.open(image_path).convert("RGB")
        
        messages = [
            {   
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                    },
                    {"type": "text", "text": user_prompt},
                ],
            }
        ]
        text = PROCESSOR.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = PROCESSOR(text=text, images=image, return_tensors="pt").to(VL_MODEL.device)

        with torch.no_grad():
            output = VL_MODEL.generate(**inputs, do_sample=True, top_p=1.0, top_k=50, temperature=1.0, repetition_penalty=1.05, max_new_tokens=50)
        
        response_text = PROCESSOR.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        # Clean the response to only get the newly generated part.
        cleaned_response = response_text.split("assistant")[-1].strip()
        return cleaned_response
    except Exception as e:
        print(f"Error during local VL model request: {e}")
        return "Local model request failed."