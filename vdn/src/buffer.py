"""
VDN 经验回放缓冲区
"""
import torch
import numpy as np
from collections import deque
from typing import Dict, List, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ReplayBuffer:
    """VDN经验回放缓冲区"""

    def __init__(self, capacity: int, n_agents: int, obs_dim: int, state_dim: int, device='cpu'):
        """
        初始化回放缓冲区

        Args:
            capacity: 缓冲区容量
            n_agents: 智能体数量
            obs_dim: 观测维度
            state_dim: 状态维度
            device: 计算设备
        """
        self.capacity = capacity
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.device = device

        # 使用deque提高效率
        self.buffer = deque(maxlen=capacity)

        # 动态检测实际的状态维度和观测维度
        self.actual_state_dim = None
        self.actual_obs_dims = None  # 用于处理异构观测维度

        # 统计信息
        self.total_transitions = 0
        self.episode_rewards = []

        logger.debug(f"经验回放缓冲区初始化: 容量={capacity}, 智能体数={n_agents}")

    def push(self,
             obs: Dict[str, np.ndarray],
             actions: Dict[str, int],
             rewards: Dict[str, float],
             next_obs: Dict[str, np.ndarray],
             dones: Dict[str, bool],
             global_state: Optional[np.ndarray] = None,
             next_global_state: Optional[np.ndarray] = None):
        """
        添加经验到缓冲区

        Args:
            obs: 当前观测字典 {agent_id: obs}
            actions: 动作字典 {agent_id: action}
            rewards: 奖励字典 {agent_id: reward}
            next_obs: 下一观测字典 {agent_id: obs}
            dones: 完成标志字典 {agent_id: done}
            global_state: 全局状态（VDN可选）
            next_global_state: 下一全局状态（VDN可选）
        """
        try:
            # 将字典转换为有序数组（按agent_id排序）
            agent_ids = sorted(obs.keys())

            # 检测并保存实际观测维度
            if self.actual_obs_dims is None:
                self.actual_obs_dims = [obs[agent_id].shape[0] for agent_id in agent_ids]
                logger.debug(f"检测到观测维度: {self.actual_obs_dims}")

            # 处理异构观测 - 分别存储每个智能体的观测
            obs_list = [obs[agent_id] for agent_id in agent_ids]
            next_obs_list = [next_obs[agent_id] for agent_id in agent_ids]

            # 转换为numpy数组（对于异构观测，这会是对象数组）
            try:
                obs_array = np.stack(obs_list)
                next_obs_array = np.stack(next_obs_list)
            except ValueError:
                # 异构观测维度，使用对象数组
                obs_array = np.array(obs_list, dtype=object)
                next_obs_array = np.array(next_obs_list, dtype=object)

            actions_array = np.array([actions[agent_id] for agent_id in agent_ids])
            rewards_array = np.array([rewards[agent_id] for agent_id in agent_ids])
            dones_array = np.array([dones[agent_id] for agent_id in agent_ids])

            # VDN中global_state是可选的，如果没有提供则使用obs的拼接
            if global_state is None:
                # 如果没有全局状态，使用所有智能体观测的拼接
                if isinstance(obs_array, np.ndarray) and obs_array.dtype == object:
                    # 异构观测
                    global_state = np.concatenate([obs for obs in obs_array])
                else:
                    # 同构观测
                    global_state = obs_array.flatten()
            else:
                global_state = np.asarray(global_state).flatten()

            if next_global_state is None:
                # 如果没有下一全局状态，使用所有智能体下一观测的拼接
                if isinstance(next_obs_array, np.ndarray) and next_obs_array.dtype == object:
                    # 异构观测
                    next_global_state = np.concatenate([obs for obs in next_obs_array])
                else:
                    # 同构观测
                    next_global_state = next_obs_array.flatten()
            else:
                next_global_state = np.asarray(next_global_state).flatten()

            # 动态检测并更新实际的状态维度
            if self.actual_state_dim is None:
                self.actual_state_dim = global_state.shape[0]
                logger.debug(f"检测到状态维度: {self.actual_state_dim}")

            # 对于动态状态维度，记录最大维度用于填充
            if not hasattr(self, 'max_state_dim'):
                self.max_state_dim = global_state.shape[0]
            else:
                self.max_state_dim = max(self.max_state_dim, global_state.shape[0])

            transition = {
                'obs': obs_array,
                'actions': actions_array,
                'rewards': rewards_array,
                'next_obs': next_obs_array,
                'dones': dones_array,
                'global_state': global_state,
                'next_global_state': next_global_state
            }

            self.buffer.append(transition)
            self.total_transitions += 1

        except Exception as e:
            logger.error(f"添加经验到缓冲区时发生错误: {str(e)}")
            raise

    def sample(self, batch_size: int) -> Tuple:
        """
        从缓冲区采样批量数据

        Args:
            batch_size: 批次大小

        Returns:
            批次数据元组 (obs, actions, rewards, next_obs, dones, global_states, next_global_states)
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)

        try:
            # 随机采样
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)

            # 收集批量数据
            batch_obs = []
            batch_actions = []
            batch_rewards = []
            batch_next_obs = []
            batch_dones = []
            batch_states = []
            batch_next_states = []

            for idx in indices:
                transition = self.buffer[idx]
                batch_obs.append(transition['obs'])
                batch_actions.append(transition['actions'])
                batch_rewards.append(transition['rewards'])
                batch_next_obs.append(transition['next_obs'])
                batch_dones.append(transition['dones'])
                batch_states.append(transition['global_state'])
                batch_next_states.append(transition['next_global_state'])

            # 转换为tensor - 处理异构观测
            first_obs = batch_obs[0]
            if isinstance(first_obs, np.ndarray) and first_obs.dtype == object:
                # 异构观测 - 分别处理每个智能体的观测，保持其原始维度
                obs_list_of_tensors = []
                next_obs_list_of_tensors = []

                for i in range(self.n_agents):
                    # 获取第i个智能体的所有观测
                    agent_obs = [batch_obs[b_idx][i] for b_idx in range(len(batch_obs))]
                    agent_next_obs = [batch_next_obs[b_idx][i] for b_idx in range(len(batch_next_obs))]

                    # 转换为tensor
                    agent_obs_tensor = torch.FloatTensor(np.stack(agent_obs)).to(self.device)
                    agent_next_obs_tensor = torch.FloatTensor(np.stack(agent_next_obs)).to(self.device)

                    obs_list_of_tensors.append(agent_obs_tensor)
                    next_obs_list_of_tensors.append(agent_next_obs_tensor)

                # 异构观测 - 返回列表格式，保持原始维度
                # 返回格式: [tensor_agent_0, tensor_agent_1, ...] 每个tensor形状为 [batch_size, obs_dim_i]
                obs_batch = obs_list_of_tensors
                next_obs_batch = next_obs_list_of_tensors
                batch_obs_tensor = None  # Not used for heterogeneous
                batch_next_obs_tensor = None  # Not used for heterogeneous
            else:
                # 同构观测 - 直接转换
                batch_obs_tensor = torch.FloatTensor(np.array(batch_obs)).to(self.device)
                batch_next_obs_tensor = torch.FloatTensor(np.array(batch_next_obs)).to(self.device)
                obs_batch = batch_obs_tensor  # For consistency in return statement
                next_obs_batch = batch_next_obs_tensor  # For consistency in return statement

            # 处理不同维度的状态
            state_dims = [state.shape[0] for state in batch_states]
            next_state_dims = [state.shape[0] for state in batch_next_states]

            # 总是填充到配置的state_dim，确保与网络匹配
            target_dim = self.state_dim

            # 检查是否需要填充
            needs_padding = any(dim != target_dim for dim in state_dims + next_state_dims)

            if not needs_padding:
                # 所有状态维度一致且等于目标维度，直接堆叠
                global_state_tensor = torch.FloatTensor(np.stack(batch_states)).to(self.device)
                next_global_state_tensor = torch.FloatTensor(np.stack(batch_next_states)).to(self.device)
            else:
                # 需要填充到目标维度
                padded_states = []
                padded_next_states = []

                for state, next_state in zip(batch_states, batch_next_states):
                    # 填充状态
                    padded_state = np.zeros(target_dim, dtype=state.dtype)
                    actual_dim = min(state.shape[0], target_dim)
                    padded_state[:actual_dim] = state[:actual_dim]
                    padded_states.append(padded_state)

                    # 填充下一个状态
                    padded_next_state = np.zeros(target_dim, dtype=next_state.dtype)
                    next_actual_dim = min(next_state.shape[0], target_dim)
                    padded_next_state[:next_actual_dim] = next_state[:next_actual_dim]
                    padded_next_states.append(padded_next_state)

                global_state_tensor = torch.FloatTensor(np.stack(padded_states)).to(self.device)
                next_global_state_tensor = torch.FloatTensor(np.stack(padded_next_states)).to(self.device)

            # 处理异构观测的返回格式
            if isinstance(obs_batch, list):
                # 异构观测 - 返回列表格式
                return (obs_batch,
                       torch.LongTensor(np.array(batch_actions)).to(self.device),
                       torch.FloatTensor(np.array(batch_rewards)).to(self.device),
                       next_obs_batch,
                       torch.FloatTensor(np.array(batch_dones)).to(self.device),
                       global_state_tensor,
                       next_global_state_tensor)
            else:
                # 同构观测 - 返回张量格式
                return (batch_obs_tensor,
                       torch.LongTensor(np.array(batch_actions)).to(self.device),
                       torch.FloatTensor(np.array(batch_rewards)).to(self.device),
                       batch_next_obs_tensor,
                       torch.FloatTensor(np.array(batch_dones)).to(self.device),
                       global_state_tensor,
                       next_global_state_tensor)

        except Exception as e:
            logger.error(f"采样批次数据时发生错误: {str(e)}")
            raise

    def __len__(self):
        """返回缓冲区大小"""
        return len(self.buffer)

    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()
        self.total_transitions = 0
        self.episode_rewards.clear()
        logger.debug("经验回放缓冲区已清空")

    def is_ready(self, batch_size: int) -> bool:
        """
        检查缓冲区是否准备好进行采样

        Args:
            batch_size: 所需批次大小

        Returns:
            是否准备好
        """
        return len(self.buffer) >= batch_size

    def get_buffer_info(self) -> Dict[str, Any]:
        """
        获取缓冲区信息

        Returns:
            缓冲区信息字典
        """
        return {
            'capacity': self.capacity,
            'current_size': len(self.buffer),
            'utilization': len(self.buffer) / self.capacity,
            'total_transitions': self.total_transitions,
            'n_agents': self.n_agents,
            'obs_dim': self.obs_dim,
            'state_dim': self.state_dim,
            'actual_obs_dims': self.actual_obs_dims,
            'actual_state_dim': self.actual_state_dim,
            'max_state_dim': getattr(self, 'max_state_dim', self.state_dim),
            'device': str(self.device)
        }

    def save_episode_reward(self, episode_reward: float):
        """
        保存episode奖励（用于性能跟踪）

        Args:
            episode_reward: episode总奖励
        """
        self.episode_rewards.append(episode_reward)

        # 保持最近1000个episode的奖励
        if len(self.episode_rewards) > 1000:
            self.episode_rewards = self.episode_rewards[-1000:]

    def get_recent_rewards(self, n_episodes: int = 100) -> List[float]:
        """
        获取最近的episode奖励

        Args:
            n_episodes: 返回最近的episode数量

        Returns:
            奖励列表
        """
        return self.episode_rewards[-n_episodes:] if self.episode_rewards else []

    def get_reward_statistics(self) -> Dict[str, float]:
        """
        获取奖励统计信息

        Returns:
            奖励统计字典
        """
        if not self.episode_rewards:
            return {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'count': 0
            }

        rewards = np.array(self.episode_rewards)
        return {
            'mean': float(np.mean(rewards)),
            'std': float(np.std(rewards)),
            'min': float(np.min(rewards)),
            'max': float(np.max(rewards)),
            'count': len(rewards)
        }