"""
VDN 训练器 - Value Decomposition Networks
"""
import os
import sys
import time
import logging
import numpy as np
import torch
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from .algos import VDN
from .buffer import ReplayBuffer
from .envs import create_env_wrapper
from .utils import calculate_vdn_statistics, save_training_data, create_experiment_name

# 创建logger
logger = logging.getLogger(__name__)


class VDNTrainer:
    """VDN训练器"""

    def __init__(self, config: Dict[str, Any], device='cpu'):
        """
        初始化VDN训练器

        Args:
            config: 配置字典
            device: 计算设备
        """
        self.config = config
        self.device = device

        # 创建环境
        self.env = create_env_wrapper(config)
        self.env_info = self.env.get_env_info()

        # 检查是否有异构观测维度
        self.heterogeneous_obs = len(set(self.env_info['obs_dims'])) > 1

        # 创建VDN算法
        self.vdn = VDN(self.env_info, config, device)

        # 创建经验回放缓冲区
        self.buffer = ReplayBuffer(
            capacity=config['training']['buffer_size'],
            n_agents=self.env_info['n_agents'],
            obs_dim=max(self.env_info['obs_dims']),
            state_dim=self.env_info['global_state_dim'],
            device=device
        )

        # 训练参数
        self.total_episodes = config['training']['total_episodes']
        self.batch_size = config['training']['batch_size']
        self.warmup_episodes = config['training'].get('warmup_episodes', 0)
        self.eval_interval = config['training']['eval_interval']
        self.save_interval = config['training']['save_interval']

        # 探索参数
        self.epsilon_start = config['exploration']['epsilon_start']
        self.epsilon_end = config['exploration']['epsilon_end']
        self.epsilon_decay = config['exploration']['epsilon_decay']
        self.current_epsilon = self.epsilon_start

        # 训练状态
        self.episode_count = 0
        self.training_steps = 0

        # 记录指标
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        self.eval_episodes = []
        self.eval_rewards = []
        self.epsilon_history = []

        # 实验信息
        self.experiment_name = create_experiment_name(
            self.env.env_name,
            self.env.difficulty,
            "VDN"
        )

        logger.info(f"VDN训练器初始化完成: {self.experiment_name}")

    def train(self) -> List[float]:
        """
        执行VDN训练

        Returns:
            episode_rewards: 所有episode的奖励列表
        """
        logger.info("开始VDN训练...")
        start_time = time.time()

        try:
            for episode in range(self.total_episodes):
                # 运行一个episode
                episode_reward, episode_length = self._run_episode()

                # 记录结果
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                self.epsilon_history.append(self.current_epsilon)

                # 更新探索率
                self._update_epsilon()

                # 训练网络
                if self.episode_count >= self.warmup_episodes:
                    self._train_network()

                # 定期评估
                if episode % self.eval_interval == 0 and episode > 0:
                    eval_reward = self.evaluate(num_episodes=10)
                    self.eval_episodes.append(episode)
                    self.eval_rewards.append(eval_reward)
                    logger.info(f"Episode {episode}: 评估奖励 = {eval_reward:.2f}")

                # 定期保存
                if episode % self.save_interval == 0 and episode > 0:
                    self._save_checkpoint(episode)

                # 记录训练信息
                if episode % 10 == 0 or episode == self.total_episodes - 1:
                    recent_rewards = self.episode_rewards[-min(100, len(self.episode_rewards)):]
                    avg_reward = np.mean(recent_rewards)
                    stats = calculate_vdn_statistics(self.episode_rewards)

                    logger.info(
                        f"Episode {episode:5d}/{self.total_episodes} | "
                        f"奖励: {episode_reward:7.2f} | "
                        f"平均: {avg_reward:7.2f} | "
                        f"最佳: {stats['max']:7.2f} | "
                        f"步数: {episode_length:4d} | "
                        f"ε: {self.current_epsilon:.3f} | "
                        f"缓冲区: {len(self.buffer):5d}"
                    )

                self.episode_count += 1

            # 训练完成
            training_time = time.time() - start_time
            logger.info(f"训练完成! 总时间: {training_time/3600:.2f}小时")

            return self.episode_rewards

        except KeyboardInterrupt:
            logger.warning("训练被用户中断")
            return self.episode_rewards
        except Exception as e:
            logger.error(f"训练过程中发生错误: {str(e)}")
            raise

    def _run_episode(self) -> Tuple[float, int]:
        """
        运行一个episode

        Returns:
            episode_reward: episode总奖励
            episode_length: episode长度
        """
        try:
            # 重置环境
            obs, info = self.env.reset()
            hidden_states = None

            episode_reward = 0.0
            episode_length = 0

            # Episode循环
            while True:
                # 转换观测为张量
                if self.heterogeneous_obs:
                    # 异构观测维度 - 使用列表形式
                    obs_list = [
                        torch.FloatTensor(obs[agent_id]).to(self.device)
                        for agent_id in self.env.agent_ids
                    ]
                else:
                    # 同构观测维度 - 可以堆叠
                    obs_tensor = torch.stack([
                        torch.FloatTensor(obs[agent_id]) for agent_id in self.env.agent_ids
                    ]).to(self.device)

                # 选择动作
                if self.heterogeneous_obs:
                    actions, hidden_states = self.vdn.select_actions_heterogeneous(
                        obs_list, hidden_states, self.current_epsilon
                    )
                else:
                    actions, hidden_states = self.vdn.select_actions(
                        obs_tensor, hidden_states, self.current_epsilon
                    )
                action_dict = {
                    agent_id: actions[i].item()
                    for i, agent_id in enumerate(self.env.agent_ids)
                }

                # 执行动作
                next_obs, rewards, dones, next_info = self.env.step(action_dict)

                # 计算总奖励
                if isinstance(rewards, dict):
                    total_reward = sum(rewards.values())
                else:
                    total_reward = rewards

                episode_reward += total_reward
                episode_length += 1

                # 添加经验到缓冲区
                self.buffer.push(
                    obs=obs,
                    actions=action_dict,
                    rewards=rewards,
                    next_obs=next_obs,
                    dones=dones,
                    global_state=info['global_state'],
                    next_global_state=next_info['global_state']
                )

                # 更新状态
                obs = next_obs
                info = next_info

                # 检查是否结束
                if isinstance(dones, dict):
                    episode_done = any(dones.values())
                else:
                    episode_done = dones

                if episode_done:
                    break

            # 保存episode奖励到缓冲区
            self.buffer.save_episode_reward(episode_reward)

            return episode_reward, episode_length

        except Exception as e:
            logger.error(f"运行episode时发生错误: {str(e)}")
            raise

    def _train_network(self):
        """训练VDN网络"""
        try:
            # 检查缓冲区是否足够
            if not self.buffer.is_ready(self.batch_size):
                return

            # 采样批次
            batch = self.buffer.sample(self.batch_size)

            # 执行训练步骤
            metrics = self.vdn.train_step(batch)

            # 记录损失
            if 'loss' in metrics:
                self.losses.append(metrics['loss'])

            self.training_steps += 1

        except Exception as e:
            logger.error(f"训练网络时发生错误: {str(e)}")
            raise

    def _update_epsilon(self):
        """更新探索率（与QMIX一致的指数衰减）"""
        # 指数衰减（与QMIX一致）
        self.current_epsilon = max(self.epsilon_end, self.current_epsilon * self.epsilon_decay)

    def _save_checkpoint(self, episode: int):
        """保存检查点（与QMIX格式一致）"""
        try:
            # 创建目录
            os.makedirs('checkpoints', exist_ok=True)

            # 使用QMIX命名格式保存模型
            checkpoint_path = f"checkpoints/vdn_episode_{episode}.pt"
            self.vdn.save_model(checkpoint_path)

            logger.debug(f"VDN检查点已保存: episode {episode}")

        except Exception as e:
            logger.warning(f"保存检查点时发生错误: {str(e)}")

    def evaluate(self, num_episodes: int = 20) -> float:
        """
        评估当前策略

        Args:
            num_episodes: 评估episodes数量

        Returns:
            平均评估奖励
        """
        try:
            logger.info(f"开始评估 ({num_episodes} episodes)...")

            # 设置为评估模式
            self.vdn.set_training_mode(False)

            eval_rewards = []

            for episode in range(num_episodes):
                result = self.vdn.evaluate_episode(
                    self.env, max_steps=None
                )
                if len(result) == 3:
                    episode_reward, episode_length, metrics = result
                else:
                    episode_reward, episode_length = result

                eval_rewards.append(episode_reward)

            # 恢复训练模式
            self.vdn.set_training_mode(True)

            avg_reward = np.mean(eval_rewards)
            std_reward = np.std(eval_rewards)

            logger.info(f"评估完成: 平均奖励 = {avg_reward:.2f} ± {std_reward:.2f}")

            return avg_reward

        except Exception as e:
            logger.error(f"评估时发生错误: {str(e)}")
            raise

    def get_training_metrics(self) -> Dict[str, Any]:
        """
        获取训练指标

        Returns:
            训练指标字典
        """
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'losses': self.losses,
            'eval_episodes': self.eval_episodes,
            'eval_rewards': self.eval_rewards,
            'epsilon_history': self.epsilon_history
        }

    def save_training_data(self, save_dir: str = 'checkpoints') -> str:
        """
        保存训练数据（与QMIX格式完全一致）

        Args:
            save_dir: 保存目录

        Returns:
            保存的文件路径
        """
        try:
            # 创建目录
            os.makedirs(save_dir, exist_ok=True)

            # 生成时间戳和环境信息
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            env_name = self.config['env']['name']
            difficulty = self.config['env']['difficulty']

            # 准备数据（与QMIX格式完全一致）
            training_data = {
                'config': self.config,
                'metrics': {
                    'episode_rewards': self.episode_rewards,
                    'episode_lengths': self.episode_lengths,
                    'losses': self.losses,
                    'eval_episodes': self.eval_episodes,
                    'eval_rewards': self.eval_rewards,
                    'epsilon_history': self.epsilon_history
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

            # 保存文件（使用QMIX命名格式）
            filename = f"{env_name}_{difficulty}_training_data_{timestamp}.json"
            filepath = os.path.join(save_dir, filename)

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(training_data_converted, f, indent=2, ensure_ascii=False)

            logger.info(f"训练数据已保存到: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"保存训练数据时发生错误: {str(e)}")
            raise

    def close(self):
        """关闭训练器，释放资源"""
        try:
            if hasattr(self, 'env') and self.env is not None:
                self.env.close()
            logger.debug("VDN训练器已关闭")
        except Exception as e:
            logger.warning(f"关闭训练器时发生错误: {str(e)}")

    def __del__(self):
        """析构函数"""
        self.close()