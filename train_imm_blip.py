# ✨ 文件名建议：train_blip_with_imitation.py

import os
import glob
import time
from datetime import datetime
import random
import PIL
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
from torch.utils.data import Dataset, DataLoader
from ppo_blip import PPO, PromptGenerationEnv
import argparse
import json
class ExpertDataset(Dataset):
    def __init__(self, expert_data):
        self.expert_data = expert_data

    def __len__(self):
        return len(self.expert_data)

    def __getitem__(self, idx):
        return self.expert_data[idx]

# collate_fn 不需要修改
def collate_fn(batch, processor):
    initial_states, expert_prompts = zip(*batch)
    padding_value = processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else 0
    
    padded_prompts = nn.utils.rnn.pad_sequence(
        expert_prompts, batch_first=True, padding_value=padding_value
    )
    
    return list(initial_states), padded_prompts

def generate_sa_pairs_from_experts(image_path, num_prompts, blip_model, processor, env, pretrain_max_tokens, device):
    """
    ✨ [V6 - 最终一致版]
    通过在循环中手动调用前向传播来生成 (状态, 动作) 对。
    此函数中的 h_t 生成逻辑现在与 ActorCritic._forward_model 中的逻辑完全一致。
    """
    print(f"--- Generating {num_prompts} expert trajectories using BLIP to create (state, action) pairs ---")
    sa_pairs = []
    
    try:
        image = PIL.Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return []

    for i in range(num_prompts):
        with torch.no_grad():
            # 1. 首先，用 .generate() 生成一个完整的专家动作序列 (expert_prompt_tokens)
            gen_inputs = processor(images=image, return_tensors="pt").to(device)
            generated_ids = blip_model.generate(
                **gen_inputs, 
                max_new_tokens=pretrain_max_tokens, 
                do_sample=True, 
                temperature=1.0
            )
            expert_prompt_tokens = generated_ids[0, 1:] # 去掉开头的 BOS token

            generated_text = processor.decode(expert_prompt_tokens, skip_special_tokens=True)
            print(f"     Processing expert prompt {i+1}/{num_prompts}: '{generated_text}'")

            # 2. 然后，在循环中手动重演这个过程，以获取与PPO训练时完全一致的 h_t
            current_prompt_tokens = []
            for token_tensor in expert_prompt_tokens:
                # 准备当前时间步的输入
                step_inputs = processor(
                    images=image, 
                    text=processor.decode(current_prompt_tokens),
                    return_tensors="pt"
                ).to(device)

                # ✨✨✨ 以下代码块与 ActorCritic._forward_model 中的逻辑完全相同 ✨✨✨
                # a. 获取图像编码
                vision_outputs = blip_model.vision_model(pixel_values=step_inputs['pixel_values'])
                image_embeds = vision_outputs[0]
                image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

                # b. 调用文本解码器
                decoder_outputs = blip_model.text_decoder(
                    input_ids=step_inputs['input_ids'],
                    attention_mask=step_inputs['attention_mask'],
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_attention_mask,
                    output_hidden_states=True
                )
                
                # c. 从 .hidden_states 元组中获取最后一层的状态
                last_hidden_states = decoder_outputs.hidden_states[-1]

                # d. 提取序列最后一个token的隐藏状态作为 h_t
                h_t = last_hidden_states[:, -1, :].detach()
                # ✨✨✨ 以上代码块与 ActorCritic._forward_model 中的逻辑完全相同 ✨✨✨

                # 动作是专家序列中的当前 token
                next_token = token_tensor.to(device, dtype=torch.long)
                
                # 记录 (状态, 动作) 对
                sa_pairs.append((h_t, next_token))
                
                # 更新上下文以用于下一步
                current_prompt_tokens.append(token_tensor.item())
                
    print(f"--- (State, Action) pair generation finished. Total pairs: {len(sa_pairs)} ---")
    return sa_pairs

def test_pretrained_policy(agent, env, max_len):
    print("\n--- Testing Pre-trained Policy ---")
    agent.policy_old.eval()
    state = env.reset()
    generated_tokens = []
    with torch.no_grad():
        for _ in range(max_len):
            action, _, _ = agent.policy_old.act(state)
            state['prompt'].append(action)
            
            # ✨ 使用正确的 EOS token id
            if action == agent.policy_old.processor.tokenizer.sep_token_id:
                break
            generated_tokens.append(action) # 只添加有效 token

    prompt_text = agent.policy.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print(f"Generated Prompt: '{prompt_text}'")
    agent.policy_old.train()
    print("--- Testing Finished ---")


# ============================================================================================
# 主训练函数
# ============================================================================================
def train():
    parser = argparse.ArgumentParser(description='RL Fine-tuning of BLIP for Image Prompt Generation')
    print("============================================================================================")
    parser.add_argument('--image_dir', type=str, default="/home/mingzhel_umass_edu/inverse/LatentTracer/data/flickr30k/004.png", help='Path to the target image file.')
    parser.add_argument('--work_dir', type=str, default="./results", help='Path to the workplace.')
    parser.add_argument('--target_model_path', type=str, default="/project/pi_shiqingma_umass_edu/mingzheli/model/stable-diffusion-v1-5", help='Path to the workplace.')
 
    args = parser.parse_args()
    ####### 超参数设置 (与原脚本保持一致) #######
    env_name = "PromptGenerationEnv"
    max_ep_len = 30
    max_training_timesteps = int(5000)
    print_freq = max_ep_len * 10
    log_freq = max_ep_len * 10
    save_model_freq = int(1e4)
    update_timestep = max_ep_len * 5
    K_epochs = 4
    eps_clip = 0.2
    gamma = 0.98
    lr_actor = 1e-4
    lr_critic = 5e-4
    random_seed = 0
    log_dir = "PPO_logs_BLIP"
    os.makedirs(log_dir, exist_ok=True)
    image_dir = args.image_dir

    
    # 模仿学习超参数
    pretrain_epochs = 2000
    pretrain_lr = 3e-4
    pretrain_batch_size = 8
    num_expert_prompts = 10
    pretrain_max_tokens = 20
    
    pretrain_checkpoint_path = f"PPO_preTrained_BLIP/PromptGenerationEnv/imitation_pretrained.pth"
    checkpoint_path = f"PPO_preTrained_BLIP/PromptGenerationEnv/PPO_{env_name}_{random_seed}_{image_dir[:-4]}.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    env = PromptGenerationEnv(diffusion_model_name=args.target_model_path, image_dir=image_dir, max_prompt_length=max_ep_len)
    print(f"PPO Hyperparameters ➞ lr_actor: {lr_actor}, lr_critic: {lr_critic}, gamma: {gamma}, K_epochs: {K_epochs}, eps_clip: {eps_clip} | Pre-training Hyperparameters ➞ epochs: {pretrain_epochs}, lr: {pretrain_lr}, batch_size: {pretrain_batch_size}, num_prompts: {num_expert_prompts}, max_tokens: {pretrain_max_tokens} | Config ➞ image: {image_dir}")

    print("\n=============================== Starting Phase 1: Imitation Learning ===============================")

    # ✨ 实例化基于 BLIP 的 PPO agent
    ppo_agent_pretrain = PPO(lr_actor=pretrain_lr, lr_critic=pretrain_lr, gamma=gamma, K_epochs=K_epochs, eps_clip=eps_clip)
    
    # 冻结 MLLM，只训练 adapter
    for param in ppo_agent_pretrain.policy_old.mllm.parameters():
        param.requires_grad = False
    for param in ppo_agent_pretrain.policy_old.adapter_mlp.parameters():
        param.requires_grad = True

    blip_model_expert = ppo_agent_pretrain.policy_old.mllm
    processor_expert = ppo_agent_pretrain.policy_old.processor
    
    # 生成 (h_t, next_token) 数据集
    sa_pairs_list = generate_sa_pairs_from_experts(image_dir, num_expert_prompts, blip_model_expert, processor_expert, env, pretrain_max_tokens, device)
    
    if not sa_pairs_list:
        print("Failed to generate expert data. Exiting.")
        return

    expert_dataset = ExpertDataset(sa_pairs_list)
    expert_dataloader = DataLoader(expert_dataset, batch_size=pretrain_batch_size, shuffle=True)

    optimizer = optim.Adam(ppo_agent_pretrain.policy_old.adapter_mlp.parameters(), lr=pretrain_lr)
    loss_fn = nn.CrossEntropyLoss()

    ppo_agent_pretrain.policy_old.train() # 确保模型在训练模式
    for epoch in range(pretrain_epochs):
        epoch_loss = 0
        for h_t_batch, next_token_batch in expert_dataloader:
            optimizer.zero_grad()
            
            h_t_batch = h_t_batch.squeeze(1).to(device)
            next_token_batch = next_token_batch.to(device)
            
            # --- 高效的监督学习步骤 ---
            # 1. 将 h_t 批次通过 adapter_mlp
            adapted_h_t = ppo_agent_pretrain.policy_old.adapter_mlp(h_t_batch)
            
            # ✨ 核心修复: 使用与 ActorCritic 中完全相同的正确路径
            # 2. 将结果通过正确的输出层得到 logits
            logits = ppo_agent_pretrain.policy_old.mllm.text_decoder.cls.predictions.decoder(adapted_h_t)
            
            # 3. 计算损失
            loss = loss_fn(logits, next_token_batch)
            # 4. 反向传播和优化
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Pre-train Epoch [{epoch+1}/{pretrain_epochs}], Average Loss: {epoch_loss/len(expert_dataloader):.4f}")

    os.makedirs(os.path.dirname(pretrain_checkpoint_path), exist_ok=True)
    ppo_agent_pretrain.save(pretrain_checkpoint_path)
    print(f"--- Imitation learning finished. Pre-trained model saved to {pretrain_checkpoint_path} ---")

    test_pretrained_policy(ppo_agent_pretrain, env, max_ep_len)

    print("\n--- Cleaning up VRAM before starting RL phase ---")
    del ppo_agent_pretrain, blip_model_expert, processor_expert, expert_dataset, expert_dataloader, optimizer, sa_pairs_list
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("--- VRAM cleaned up ---")

    print("\n============================== Starting Phase 2: RL Fine-tuning ==============================")
    
# ✨ Re-instantiate the PPO agent
    ppo_agent_rl = PPO(lr_actor=lr_actor, lr_critic=lr_critic, gamma=gamma, K_epochs=K_epochs, eps_clip=eps_clip)
    
    try:
        ppo_agent_rl.load(pretrain_checkpoint_path)
        print(f"--- Successfully loaded pre-trained model from {pretrain_checkpoint_path} for RL fine-tuning ---")
    except FileNotFoundError:
        print("--- No pre-trained model found. Starting RL training from scratch. ---")

    start_time = datetime.now().replace(microsecond=0)
    print("Started RL training at (GMT): ", start_time)
    


    # <<< NEW: Initialize variables to track the best reward and prompt >>>
    best_reward = -float('inf')
    best_prompt = ""

    time_step = 0
    i_episode = 0

    while time_step <= max_training_timesteps:
        state = env.reset()
        current_ep_reward = 0
        
        for t in range(1, max_ep_len + 1):
            action = ppo_agent_rl.select_action(state)
            state, reward, done, _ = env.step(action)
            
            ppo_agent_rl.buffer.rewards.append(reward)
            ppo_agent_rl.buffer.is_terminals.append(done)
            
            time_step += 1
            current_ep_reward += reward

            if time_step % update_timestep == 0:
                ppo_agent_rl.update()

            if done:
                break
        
        # This part that decodes the prompt is the same
        prompt_text = env.processor.tokenizer.decode(state["prompt"][len(env.initial_tokens):], skip_special_tokens=True)
        print(f"Episode: {i_episode + 1} \t Timestep: {time_step} \t Reward: {current_ep_reward:.4f} \t Prompt: '{prompt_text}'")

        # <<< NEW: Check if the current episode's reward is the best so far >>>
        if current_ep_reward > best_reward:
            best_reward = current_ep_reward
            best_prompt = prompt_text
            print(f"🎉 New best reward found! Reward: {best_reward:.4f}")

        i_episode += 1

    # <<< NEW: Save the best prompt to a JSON file after training is complete >>>
    # We assume the image path is available in your 'env' or 'args' object.
    # This code will try to find it.
    image_path = args.image_dir # Fallback path


    output_filename = os.path.join(args.work_dir, "best_prompts.json")
    results = {}
    
    # Load existing data to avoid overwriting results for other images
    if os.path.exists(output_filename):
        with open(output_filename, 'r') as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                results = {} # Start with a fresh dictionary if file is corrupted

    # Update the dictionary with the result from this run
    results[image_path] = {
        "best_prompt": best_prompt,
        "best_reward": round(best_reward, 4)
    }

    # Write the updated dictionary back to the file
    with open(output_filename, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"--- Best prompt for '{image_path}' saved to {output_filename} ---")

    env.close()

    end_time = datetime.now().replace(microsecond=0)
    print("Total training time: ", end_time - start_time)
    print("============================================================================================")

if __name__ == '__main__':
    train()