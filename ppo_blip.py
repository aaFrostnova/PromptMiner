import os
import glob
import time
from datetime import datetime
import PIL
import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import gym
from diffusers import StableDiffusionPipeline, AutoPipelineForText2Image
# ✨ 导入 BLIP 相关的模型和处理器
from transformers import CLIPModel, CLIPProcessor, BlipForConditionalGeneration, BlipProcessor


# Set device
print("============================================================================================")
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to: " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to: cpu")
print("============================================================================================")


# Custom Environment
class PromptGenerationEnv(gym.Env):
    def __init__(self,
                 diffusion_model_name="/project/pi_shiqingma_umass_edu/mingzheli/model/sdxl-turbo",
                 clip_model_name="/project/pi_shiqingma_umass_edu/mingzheli/model/clip-vit-large-patch14",
                 blip_model_name="/project/pi_shiqingma_umass_edu/mingzheli/model/blip-image-captioning-large",
                 max_prompt_length=30,
                 image_dir="./image.png",
                 w_clip=0.8,
                 w_ppl=0.2):

        super(PromptGenerationEnv, self).__init__()
        self.image_path = image_dir
        self.max_prompt_length = max_prompt_length
        self.w_clip = w_clip
        self.w_ppl = w_ppl
        self.diffusion_model_name = diffusion_model_name

        # Diffusion and CLIP models
        if "sdxl" in self.diffusion_model_name:
            self.diffusion_model = AutoPipelineForText2Image.from_pretrained(diffusion_model_name).to(device)
        else:
            self.diffusion_model = StableDiffusionPipeline.from_pretrained(diffusion_model_name).to(device)
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        
        self.processor = BlipProcessor.from_pretrained(blip_model_name)
        self.action_space = gym.spaces.Discrete(self.processor.tokenizer.vocab_size)
        
        # ✨ 核心修改 1: 不再需要初始 token，prompt 从一个空列表开始
        self.initial_tokens = []

        self.observation_space = gym.spaces.Dict({
            "prompt": gym.spaces.Sequence(gym.spaces.Discrete(self.processor.tokenizer.vocab_size)),
            "image_encoding": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.clip_model.config.projection_dim,), dtype=np.float32)
        })

    def reset(self):
        # ✨ self.prompt 现在被初始化为空列表
        self.prompt = self.initial_tokens.copy()
        self.target_image = PIL.Image.open(self.image_path).convert("RGB")
        image_inputs = self.clip_processor(images=self.target_image, return_tensors="pt").to(device)
        self.image_encoding = self.clip_model.get_image_features(**image_inputs).detach().cpu().numpy()[0]

        return {
            "prompt": self.prompt.copy(),
            "image_encoding": self.image_encoding,
            "target_image": self.target_image
        }

    def step(self, action):
        self.prompt.append(action)
        done = (len(self.prompt) >= self.max_prompt_length) or (action == self.processor.tokenizer.sep_token_id)

        # 对于非终止状态，奖励为0
        reward = 0.0
        clip_similarity = 0.0
        # 仅在 episode 结束时计算最终奖励
        if done:
            prompt_text = self.processor.tokenizer.decode(self.prompt, skip_special_tokens=True)

            try:
                # --- 1. 生成图片并计算原始 CLIP 相似度 ---
                generator = torch.Generator(device=device).manual_seed(0)
                if "sdxl" in self.diffusion_model_name:
                    generated_image = self.diffusion_model(prompt_text, num_inference_steps=1, guidance_scale=0.0, generator=generator).images[0]
                else:
                    generated_image = self.diffusion_model(prompt_text, generator=generator).images[0]
                generated_inputs = self.clip_processor(images=generated_image, return_tensors="pt").to(device)
                generated_encoding = self.clip_model.get_image_features(**generated_inputs).detach().cpu().numpy()[0]
                
                clip_similarity = np.dot(generated_encoding, self.image_encoding) / (np.linalg.norm(generated_encoding) * np.linalg.norm(self.image_encoding))

                # ✨ --- 2. 根据您的分级阈值规则设置最终奖励 ---
                if clip_similarity > 0.90:
                    reward = 3.0
                elif clip_similarity > 0.80:
                    reward = 2.0
                elif clip_similarity > 0.75:
                    reward = 1.5
                elif clip_similarity > 0.70:
                    reward = 1.0
                elif clip_similarity > 0.65:
                    reward = 0.5
                else:
                    # ✨ 核心修改：当分数低于0.65时，奖励是连续的
                    # 这样做可以让 0.6 的分比 0.5 的分得到更高的奖励
                    # 例如，这里将 [0, 0.65] 区间映射到 [-1.0, -0.35]
                    reward = clip_similarity - 1.0
                
                # (可选) 增加日志，方便观察
                print(f"Done. Clip Sim: {clip_similarity:.4f}, Reward: {reward}, Prompt: '{prompt_text}'")

            except Exception as e:
                print(f"Error during image generation or CLIP calculation: {e}")
                # ✨ 如果过程中出现任何错误，也给予 -1 的惩罚
                reward = -1.0
        
        state = {
            "prompt": self.prompt.copy(),
            "image_encoding": self.image_encoding,
            "target_image": self.target_image
        }
        
        return state, reward, done, {}, clip_similarity

# Rollout Buffer (No changes needed)
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, mllm, processor):
        super(ActorCritic, self).__init__()
        self.mllm = mllm 
        self.processor = processor

        # ✨ 核心修复 1: 统一使用 768 维的文本 hidden_size
        self.hidden_dim = self.mllm.config.text_config.hidden_size  # This is 768

        print(f"Initializing ActorCritic: Consistently using hidden dimension = {self.hidden_dim}")

        # ✨ Adapter 不改变维度 (768 -> 1536 -> 768)
        self.adapter_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        ).to(device)

        # ✨ Value head 也作用于 768 维的文本状态
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1)
        ).to(device)

    def _forward_model(self, states_batch):
        """
        ✨ [V4 - 已修复 AttributeError for CausalLMOutput...]
        """
        images_batch = [s["target_image"] for s in states_batch]
        prompts_ids_batch = [s["prompt"] for s in states_batch]

        inputs = self.processor(
            images=images_batch,
            text=[self.processor.tokenizer.decode(p) for p in prompts_ids_batch],
            padding="longest",
            return_tensors="pt",
        ).to(self.mllm.device)

        # --- 手动分步前向传播以获取正确的 state ---
        
        vision_outputs = self.mllm.vision_model(pixel_values=inputs['pixel_values'])
        image_embeds = vision_outputs[0]
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        # ✨ 核心修复 1: 必须向解码器明确请求 hidden_states
        decoder_outputs = self.mllm.text_decoder(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            output_hidden_states=True  # <-- 关键！
        )
        
        # ✨ 核心修复 2: 从 .hidden_states 元组中获取最后一层的状态
        last_hidden_states = decoder_outputs.hidden_states[-1]
        
        # --- 逻辑不变 ---
        batch_size = len(states_batch)
        sequence_lengths = inputs['attention_mask'].sum(dim=1)
        
        h_t = last_hidden_states[torch.arange(batch_size, device=self.mllm.device), sequence_lengths - 1]
        
        adapted_h_t = self.adapter_mlp(h_t)
        policy_logits = self.mllm.text_decoder.cls.predictions.decoder(adapted_h_t)

        return policy_logits, h_t
    
    def act(self, state):
        policy_logits, h_t = self._forward_model([state])
        with torch.no_grad():
            state_val = self.value_head(h_t)
            dist = Categorical(logits=policy_logits)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
        return action.item(), action_logprob.item(), state_val.item()

    def evaluate(self, states, actions):
        policy_logits, h_t = self._forward_model(states)
        dist = Categorical(logits=policy_logits)
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        state_values = self.value_head(h_t).squeeze(-1)
        return action_logprobs, state_values, dist_entropy


# PPO Class (No changes needed, as it relies on ActorCritic and Env)
class PPO:
    def __init__(self, lr_actor=1e-5, lr_critic=1e-5, gamma=0.99, K_epochs=4, eps_clip=0.2):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.buffer = RolloutBuffer()
        
        blip_model_name = "/project/pi_shiqingma_umass_edu/mingzheli/model/blip-image-captioning-large"
        mllm = BlipForConditionalGeneration.from_pretrained(blip_model_name).to(device)
        
        mllm.gradient_checkpointing_enable()
        for param in mllm.parameters():
            param.requires_grad = False
        
        processor = BlipProcessor.from_pretrained(blip_model_name)
        
        self.policy = ActorCritic(mllm=mllm, processor=processor).to(device)
        self.policy_old = ActorCritic(mllm=mllm, processor=processor).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer = torch.optim.Adam([
            {'params': self.policy.adapter_mlp.parameters(), 'lr': lr_actor},
            {'params': self.policy.value_head.parameters(), 'lr': lr_critic}
        ])
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        with torch.no_grad():
            action, action_logprob, state_val = self.policy_old.act(state)
        
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)
        
        return action

    def update(self):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states = self.buffer.states
        old_actions = self.buffer.actions
        old_logprobs = torch.tensor(self.buffer.logprobs, dtype=torch.float32).to(device)
        old_state_values = torch.tensor(self.buffer.state_values, dtype=torch.float32).to(device)
        advantages = (rewards - old_state_values).detach()

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            ratios = torch.exp(logprobs - old_logprobs)
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            print(f"value_loss: {self.MseLoss(state_values, rewards).item()}")
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

    def save(self, checkpoint_path):
        state_to_save = {
            'adapter_mlp_state_dict': self.policy_old.adapter_mlp.state_dict(),
            'value_head_state_dict': self.policy_old.value_head.state_dict()
        }
        torch.save(state_to_save, checkpoint_path)
        print(f"Efficiently saved adapter_mlp and value_head weights to {checkpoint_path}")

    def load(self, checkpoint_path):
        device = next(self.policy.parameters()).device
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        self.policy.adapter_mlp.load_state_dict(checkpoint['adapter_mlp_state_dict'])
        self.policy_old.adapter_mlp.load_state_dict(checkpoint['adapter_mlp_state_dict'])
        
        self.policy.value_head.load_state_dict(checkpoint['value_head_state_dict'])
        self.policy_old.value_head.load_state_dict(checkpoint['value_head_state_dict'])
        
        print(f"Loaded adapter_mlp and value_head weights from {checkpoint_path}")


# Training Loop
def main():
    env = PromptGenerationEnv(image_dir="./image.png")
    ppo = PPO()
    max_episodes = 1000
    update_timestep = 200
    timestep = 0
    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = ppo.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            
            ppo.buffer.rewards.append(reward)
            ppo.buffer.is_terminals.append(done)
            
            state = next_state
            episode_reward += reward
            timestep += 1
            
            if timestep >= update_timestep:
                ppo.update()
                timestep = 0
        
        # ✨ 解码逻辑现在也简化了
        prompt_text = ppo.policy.processor.tokenizer.decode(state["prompt"], skip_special_tokens=True)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.4f}, Prompt = '{prompt_text}'")

        if (episode + 1) % 100 == 0:
            ppo.save(f"ppo_checkpoint_episode_{episode + 1}.pth")


if __name__ == "__main__":
    main()