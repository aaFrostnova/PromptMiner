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

# collate_fn
def collate_fn(batch, processor):
    initial_states, expert_prompts = zip(*batch)
    padding_value = processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else 0
    
    padded_prompts = nn.utils.rnn.pad_sequence(
        expert_prompts, batch_first=True, padding_value=padding_value
    )
    
    return list(initial_states), padded_prompts

def evaluate_prompt_tokens(env, token_ids, max_len):
    state = env.reset()
    final_clip_similarity = float("-inf")
    done = False

    for tok in token_ids[:max_len]:
        state, r, done, _, clip_similarity = env.step(int(tok))
        if clip_similarity is not None:
            final_clip_similarity = float(clip_similarity)
        if done:
            break

    gen_tokens = state["prompt"][len(getattr(env, "initial_tokens", [])):]
    decoded_text = env.processor.tokenizer.decode(gen_tokens, skip_special_tokens=True)
    return final_clip_similarity, decoded_text, done

def generate_sa_pairs_from_experts(image_path, num_prompts, blip_model, processor, env, pretrain_max_tokens, device):
    print(f"--- Generating {num_prompts} expert trajectories using BLIP to create (state, action) pairs ---")
    sa_pairs = []
    expert_sequences = [] 
    
    try:
        image = PIL.Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return [], []

    for i in range(num_prompts):
        with torch.no_grad():
            gen_inputs = processor(images=image, return_tensors="pt").to(device)
            generated_ids = blip_model.generate(
                **gen_inputs, 
                max_new_tokens=pretrain_max_tokens, 
                do_sample=True, 
                temperature=1.0
            )
            expert_prompt_tokens = generated_ids[0, 1:]  

            generated_text = processor.decode(expert_prompt_tokens, skip_special_tokens=True)
            print(f"     Processing expert prompt {i+1}/{num_prompts}: '{generated_text}'")

            expert_sequences.append([int(t.item()) for t in expert_prompt_tokens])


            current_prompt_tokens = []
            for token_tensor in expert_prompt_tokens:
                step_inputs = processor(
                    images=image, 
                    text=processor.decode(current_prompt_tokens),
                    return_tensors="pt"
                ).to(device)

                vision_outputs = blip_model.vision_model(pixel_values=step_inputs['pixel_values'])
                image_embeds = vision_outputs[0]
                image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

                decoder_outputs = blip_model.text_decoder(
                    input_ids=step_inputs['input_ids'],
                    attention_mask=step_inputs['attention_mask'],
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_attention_mask,
                    output_hidden_states=True
                )
                last_hidden_states = decoder_outputs.hidden_states[-1]
                h_t = last_hidden_states[:, -1, :].detach()
                # =======================================

                next_token = token_tensor.to(device, dtype=torch.long)
                sa_pairs.append((h_t, next_token))
                current_prompt_tokens.append(token_tensor.item())
                
    print(f"--- (State, Action) pair generation finished. Total pairs: {len(sa_pairs)} ---")
    return sa_pairs, expert_sequences

def test_pretrained_policy(agent, env, max_len):
    print("\n--- Testing Pre-trained Policy ---")
    agent.policy_old.eval()
    state = env.reset()
    generated_tokens = []
    with torch.no_grad():
        for _ in range(max_len):
            action, _, _ = agent.policy_old.act(state)
            state['prompt'].append(action)
            if action == agent.policy_old.processor.tokenizer.sep_token_id:
                break
            generated_tokens.append(action)

    prompt_text = agent.policy.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print(f"Generated Prompt: '{prompt_text}'")
    agent.policy_old.train()
    print("--- Testing Finished ---")


def train():
    parser = argparse.ArgumentParser(description='RL Fine-tuning of BLIP for Image Prompt Generation')
    print("============================================================================================")
    parser.add_argument('--image_dir', type=str, help='Path to the target image file.')
    parser.add_argument('--work_dir', type=str, default="./results", help='Path to the workplace.')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--target_model_path', type=str, help='Target diffusion model path.')
    parser.add_argument('--image_range', type=str, default="0_10", help='image range')
    parser.add_argument('--step', type=int, default=2000, help='optimization step')
    args = parser.parse_args()

    # --- Set Seeds for Reproducibility ---
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    env_name = "PromptGenerationEnv"
    max_ep_len = 30
    max_training_timesteps = args.step
    print_freq = max_ep_len * 10
    log_freq = max_ep_len * 10
    save_model_freq = int(1e4)
    update_timestep = max_ep_len * 5
    K_epochs = 4
    eps_clip = 0.2
    gamma = 0.98
    lr_actor = 1e-4
    lr_critic = 5e-4
    random_seed = 42
    log_dir = "PPO_logs_BLIP"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(args.work_dir, exist_ok=True)
    image_dir = args.image_dir

    pretrain_epochs = 2000
    pretrain_lr = 3e-4
    pretrain_batch_size = 8
    num_expert_prompts = 10
    pretrain_max_tokens = 20
    
    pretrain_checkpoint_path = f"PPO_preTrained_BLIP/PromptGenerationEnv/imitation_pretrained_{args.image_range}.pth"
    checkpoint_path = f"PPO_preTrained_BLIP/PromptGenerationEnv/PPO_{env_name}_{random_seed}_{image_dir[:-4]}.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    env = PromptGenerationEnv(diffusion_model_name=args.target_model_path, image_dir=image_dir, max_prompt_length=max_ep_len, gamma=gamma, args=args)
    print(f"PPO Hyperparameters ➞ lr_actor: {lr_actor}, lr_critic: {lr_critic}, gamma: {gamma}, K_epochs: {K_epochs}, eps_clip: {eps_clip} | "
          f"Pre-training Hyperparameters ➞ epochs: {pretrain_epochs}, lr: {pretrain_lr}, batch_size: {pretrain_batch_size}, "
          f"num_prompts: {num_expert_prompts}, max_tokens: {pretrain_max_tokens} | Config ➞ image: {image_dir}")

    print("\n=============================== Starting Phase 1: Imitation Learning ===============================")

    ppo_agent_pretrain = PPO(lr_actor=pretrain_lr, lr_critic=pretrain_lr, gamma=gamma, K_epochs=K_epochs, eps_clip=eps_clip)
    
    for param in ppo_agent_pretrain.policy_old.mllm.parameters():
        param.requires_grad = False
    for param in ppo_agent_pretrain.policy_old.adapter_mlp.parameters():
        param.requires_grad = True

    blip_model_expert = ppo_agent_pretrain.policy_old.mllm
    processor_expert = ppo_agent_pretrain.policy_old.processor
    
    sa_pairs_list, expert_sequences = generate_sa_pairs_from_experts(
        image_dir, num_expert_prompts, blip_model_expert, processor_expert, env, pretrain_max_tokens, device
    )
    
    if not sa_pairs_list:
        print("Failed to generate expert data. Exiting.")
        return

    expert_dataset = ExpertDataset(sa_pairs_list)
    expert_dataloader = DataLoader(expert_dataset, batch_size=pretrain_batch_size, shuffle=True)

    optimizer = optim.Adam(ppo_agent_pretrain.policy_old.adapter_mlp.parameters(), lr=pretrain_lr)
    loss_fn = nn.CrossEntropyLoss()

    ppo_agent_pretrain.policy_old.train()  
    for epoch in range(pretrain_epochs):
        epoch_loss = 0
        for h_t_batch, next_token_batch in expert_dataloader:
            optimizer.zero_grad()
            
            h_t_batch = h_t_batch.squeeze(1).to(device)
            next_token_batch = next_token_batch.to(device)
            
            adapted_h_t = ppo_agent_pretrain.policy_old.adapter_mlp(h_t_batch)
            logits = ppo_agent_pretrain.policy_old.mllm.text_decoder.cls.predictions.decoder(adapted_h_t)
            loss = loss_fn(logits, next_token_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Pre-train Epoch [{epoch+1}/{pretrain_epochs}], Average Loss: {epoch_loss/len(expert_dataloader):.4f}")

    os.makedirs(os.path.dirname(pretrain_checkpoint_path), exist_ok=True)
    ppo_agent_pretrain.save(pretrain_checkpoint_path)
    print(f"--- Imitation learning finished. Pre-trained model saved to {pretrain_checkpoint_path} ---")

    print("\n--- Evaluating expert prompts generated during imitation data collection ---")
    best_il_clip_similarity = float("-inf")
    best_il_tokens = None
    best_il_text = ""
    for i, seq in enumerate(expert_sequences):
        clip_sim, txt, done = evaluate_prompt_tokens(env, seq, max_ep_len)
        print(f"  Expert[{i+1}/{len(expert_sequences)}] clip_sim={clip_sim:.4f} | '{txt}'")
        if clip_sim > best_il_clip_similarity:
            best_il_clip_similarity = clip_sim
            best_il_tokens = seq
            best_il_text = txt
    print(f"--- Best imitation prompt (by CLIP) = {best_il_clip_similarity:.4f} | '{best_il_text}' ---")

    test_pretrained_policy(ppo_agent_pretrain, env, max_ep_len)

    print("\n--- Cleaning up VRAM before starting RL phase ---")
    del ppo_agent_pretrain, blip_model_expert, processor_expert, expert_dataset, expert_dataloader, optimizer, sa_pairs_list, expert_sequences
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("--- VRAM cleaned up ---")

    print("\n============================== Starting Phase 2: RL Fine-tuning ==============================")
    
    ppo_agent_rl = PPO(lr_actor=lr_actor, lr_critic=lr_critic, gamma=gamma, K_epochs=K_epochs, eps_clip=eps_clip)
    
    try:
        ppo_agent_rl.load(pretrain_checkpoint_path)
        print(f"--- Successfully loaded pre-trained model from {pretrain_checkpoint_path} for RL fine-tuning ---")
    except FileNotFoundError:
        print("--- No pre-trained model found. Starting RL training from scratch. ---")

    start_time = datetime.now().replace(microsecond=0)
    print("Started RL training at (GMT): ", start_time)

    rl_best_reward = float("-inf")
    rl_best_clip_similarity = float("-inf")
    rl_best_prompt = ""
    rl_best_tokens = None

    time_step = 0
    i_episode = 0

    while time_step <= max_training_timesteps:
        state = env.reset()
        current_ep_reward = 0.0
        current_ep_clip_similarity = float("-inf")
        
        for t in range(1, max_ep_len + 1):
            action = ppo_agent_rl.select_action(state)
            # state, reward, done, info, clip_similarity
            state, reward, done, _, clip_similarity = env.step(action)
            
            ppo_agent_rl.buffer.rewards.append(reward)
            ppo_agent_rl.buffer.is_terminals.append(done)
            
            time_step += 1
            current_ep_reward += float(reward)
            if clip_similarity is not None:
                current_ep_clip_similarity = float(clip_similarity)

            if time_step % update_timestep == 0:
                ppo_agent_rl.update()

            if done:
                break
        
        prompt_text = env.processor.tokenizer.decode(state["prompt"][len(getattr(env, "initial_tokens", [])):], skip_special_tokens=True)
        print(f"Episode: {i_episode + 1} \t Timestep: {time_step} \t Reward: {current_ep_reward:.4f} \t CLIP: {current_ep_clip_similarity:.4f} \t Prompt: '{prompt_text}'")

        if current_ep_reward > rl_best_reward:
            rl_best_reward = current_ep_reward
        if current_ep_clip_similarity > rl_best_clip_similarity:
            rl_best_clip_similarity = current_ep_clip_similarity
            rl_best_prompt = prompt_text
            rl_best_tokens = [int(t) for t in state["prompt"][len(getattr(env, "initial_tokens", [])):] ]
            print(f"🎉 New RL best CLIP similarity! CLIP: {rl_best_clip_similarity:.4f}")

        i_episode += 1

    print("\n============================== Final Prompt Selection (by CLIP similarity) ==============================")
    print(f"Best IL CLIP={best_il_clip_similarity:.4f} | prompt='{best_il_text}'")
    print(f"Best RL CLIP={rl_best_clip_similarity:.4f} | prompt='{rl_best_prompt}'")

    if best_il_clip_similarity > rl_best_clip_similarity:
        final_choice = "IL"
        final_clip = best_il_clip_similarity
        final_prompt = best_il_text
        final_tokens = best_il_tokens
    else:
        final_choice = "RL"
        final_clip = rl_best_clip_similarity
        final_prompt = rl_best_prompt
        final_tokens = rl_best_tokens

    print(f"--- Final Choice: {final_choice} ---")
    print(f"Final Prompt (CLIP={final_clip:.4f}): '{final_prompt}'")

    image_path = args.image_dir
    output_filename = os.path.join(args.work_dir, "best_prompts.json")
    results = {}
    
    if os.path.exists(output_filename):
        with open(output_filename, 'r') as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                results = {}

    results[image_path] = {
        "final_choice": final_choice,                
        "best_prompt": final_prompt,
        "final_clip_similarity": round(float(final_clip), 6),
        "il": {
            "best_prompt": best_il_text,
            "best_clip_similarity": round(float(best_il_clip_similarity), 6),
        },
        "rl": {
            "best_prompt": rl_best_prompt,
            "best_clip_similarity": round(float(rl_best_clip_similarity), 6),
            "best_episode_reward_observed": round(float(rl_best_reward), 6)  
        },
        "tokens": final_tokens if isinstance(final_tokens, list) else None,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(output_filename, 'w') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"--- Final prompt for '{image_path}' saved to {output_filename} ---")

    env.close()

    end_time = datetime.now().replace(microsecond=0)
    print("Total training time: ", end_time - start_time)
    print("============================================================================================")

if __name__ == '__main__':
    train()
