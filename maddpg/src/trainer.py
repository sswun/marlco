"""
MADDPG 训练器
"""
import time
import numpy as np
import torch
from typing import Dict, Any, List, Tuple
from .envs import EnvWrapper
from .models import MADDPGNetworks
from .algos import MADDPG
from .buffer import ReplayBuffer
from .utils import get_device, to_tensor


class MADDPGTrainer:
    """MADDPG训练器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = get_device()
        
        # 创建环境
        self.env = EnvWrapper(config)
        self.env_info = self.env.get_env_info()
        
        print(f"🌍 Environment: {config['env']['name']}")
        print(f"   Agents: {self.env_info['n_agents']}")
        print(f"   Obs dims: {self.env_info['obs_dims']}")
        print(f"   Action dims: {self.env_info['act_dims']}")
        print(f"   Device: {self.device}")
        
        # 创建缓冲区
        self.buffer = ReplayBuffer(
            capacity=config['training']['buffer_size'],
            n_agents=self.env_info['n_agents'],
            obs_dim=self.env_info['obs_dims'][0],
            state_dim=0,  # MADDPG不需要全局状态
            device=self.device
        )

        # 创建网络
        self.networks = MADDPGNetworks(self.env_info, config, self.device)

        # 创建算法
        self.algorithm = MADDPG(self.networks, config, self.device)
        
        # 训练参数
        self.total_episodes = config['training']['total_episodes']
        self.batch_size = config['training']['batch_size']
        self.warmup_episodes = config['training']['warmup_episodes']
        self.eval_interval = config['training']['eval_interval']
        self.save_interval = config['training']['save_interval']
        
        # 探索参数
        self.noise_scale = config['exploration']['noise_scale']
        self.noise_decay = config['exploration']['noise_decay']
        self.noise_min = config['exploration']['noise_min']
        
        # 统计信息
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        self.eval_episodes = []
        self.eval_rewards = []
        self.noise_history = []
        
    def train(self) -> List[float]:
        """主训练循环"""
        print(f"\n🚀 开始MADDPG训练...")
        print(f"   总episodes: {self.total_episodes}")
        print(f"   预热episodes: {self.warmup_episodes}")
        print(f"   批大小: {self.batch_size}")
        
        start_time = time.time()
        
        for episode in range(self.total_episodes):
            # 收集一个episode的经验
            episode_reward, episode_length = self.collect_episode()
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            # 训练网络
            if len(self.buffer) >= self.batch_size and episode >= self.warmup_episodes:
                loss_info = self.train_step()
                self.losses.append(loss_info['total_loss'])
            
            # 更新噪声尺度
            self.noise_scale = max(self.noise_min, self.noise_scale * self.noise_decay)
            self.noise_history.append(self.noise_scale)

            # 日志和评估
            if episode % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_length = np.mean(self.episode_lengths[-100:])
                print(f"Episode {episode:6d} | Avg Reward: {avg_reward:6.2f} | "
                      f"Avg Length: {avg_length:4.1f} | Noise: {self.noise_scale:.3f} | "
                      f"Buffer: {len(self.buffer):5d}")

            # 评估
            if episode % self.eval_interval == 0 and episode > 0:
                eval_reward = self.evaluate()
                self.eval_episodes.append(episode)
                self.eval_rewards.append(eval_reward)
                print(f"🎯 Evaluation at episode {episode}: {eval_reward:.2f}")
            
            # 保存模型
            if episode % self.save_interval == 0 and episode > 0:
                self.save_checkpoint(episode)
        
        total_time = time.time() - start_time
        print(f"\n✅ 训练完成! 总时间: {total_time:.1f}s")
        
        return self.episode_rewards
    
    def collect_episode(self) -> Tuple[float, int]:
        """收集一个episode的经验"""
        obs, _ = self.env.reset()

        episode_reward = 0
        step_count = 0
        
        while True:
            # 转换观测为tensor
            obs_tensor = {}
            for agent_id, agent_obs in obs.items():
                obs_tensor[agent_id] = to_tensor(agent_obs, device=self.device)
            
            # 获取可用动作（如果环境支持）
            avail_actions = None
            if hasattr(self.env, 'get_avail_actions'):
                avail_actions = {}
                for agent_id in obs.keys():
                    avail_actions[agent_id] = self.env.get_avail_actions(agent_id)
            
            # 选择动作（带探索噪声）
            actions = self.algorithm.select_actions(obs_tensor, self.noise_scale, avail_actions)
            
            # 环境交互
            next_obs, rewards, dones, infos = self.env.step(actions)
            
            # 存储经验（MADDPG不需要全局状态）
            self.buffer.push(
                obs, actions, rewards, next_obs, dones,
                np.zeros(1), np.zeros(1)  # 传入空的全局状态
            )
            
            # 更新状态
            obs = next_obs
            episode_reward += sum(rewards.values())
            step_count += 1
            
            # 检查结束条件
            if all(dones.values()) or step_count >= 200:  # 最大步数限制
                break
        
        return episode_reward, step_count
    
    def train_step(self) -> Dict[str, float]:
        """执行一步训练"""
        batch = self.buffer.sample(self.batch_size)
        loss_info = self.algorithm.update(batch)
        return loss_info
    
    def evaluate(self, num_episodes: int = 10) -> float:
        """评估性能"""
        total_rewards = []
        
        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            step_count = 0
            
            while True:
                # 转换观测为tensor
                obs_tensor = {}
                for agent_id, agent_obs in obs.items():
                    obs_tensor[agent_id] = to_tensor(agent_obs, device=self.device)
                
                # 获取可用动作（如果环境支持）
                avail_actions = None
                if hasattr(self.env, 'get_avail_actions'):
                    avail_actions = {}
                    for agent_id in obs.keys():
                        avail_actions[agent_id] = self.env.get_avail_actions(agent_id)
                
                # 贪婪策略（无探索）
                actions = self.algorithm.select_actions(obs_tensor, noise_scale=0.0, avail_actions=avail_actions)
                
                # 环境交互
                next_obs, rewards, dones, _ = self.env.step(actions)
                
                obs = next_obs
                episode_reward += sum(rewards.values())
                step_count += 1
                
                if all(dones.values()) or step_count >= 200:
                    break
            
            total_rewards.append(episode_reward)
        
        return float(np.mean(total_rewards))
    
    def save_checkpoint(self, episode: int):
        """保存检查点"""
        checkpoint_path = f"checkpoints/maddpg_episode_{episode}.pt"
        self.algorithm.save(checkpoint_path)
        print(f"💾 模型已保存: {checkpoint_path}")
    
    def save_training_data(self, save_dir: str = "checkpoints"):
        """
        保存训练数据
        
        Args:
            save_dir: 保存目录
        """
        import os
        import json
        from datetime import datetime
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        env_name = self.config['env']['name']
        difficulty = self.config['env']['difficulty']
        
        # 准备数据
        training_data = {
            'config': self.config,
            'metrics': {
                'episode_rewards': self.episode_rewards,
                'episode_lengths': self.episode_lengths,
                'losses': self.losses,
                'eval_episodes': self.eval_episodes,
                'eval_rewards': self.eval_rewards,
                'noise_history': self.noise_history
            },
            'environment': {
                'name': env_name,
                'difficulty': difficulty,
                'n_agents': self.env_info['n_agents'],
                'obs_dims': self.env_info['obs_dims'],
                'act_dims': self.env_info['act_dims']
            },
            'timestamp': timestamp,
            'total_episodes': len(self.episode_rewards)
        }
        
        # 保存为JSON文件
        filename = f"{env_name}_{difficulty}_training_data_{timestamp}.json"
        filepath = os.path.join(save_dir, filename)
        
        # 转换numpy类型为Python原生类型
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj

        training_data_converted = convert_numpy(training_data)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(training_data_converted, f, indent=2, ensure_ascii=False)

        print(f"💾 训练数据已保存: {filepath}")
        return filepath
    
    def get_training_metrics(self):
        """Get all training metrics for plotting"""
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'losses': self.losses,
            'eval_episodes': self.eval_episodes,
            'eval_rewards': self.eval_rewards,
            'epsilon_history': self.noise_history  # 使用noise_history替代epsilon
        }

    def close(self):
        """关闭环境"""
        self.env.close()
