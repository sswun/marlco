"""
经验回放缓冲区
"""
import torch
import numpy as np
from collections import deque
from typing import Dict, List, Any, Tuple


class ReplayBuffer:
    """QMIX经验回放缓冲区"""

    def __init__(self, capacity: int, n_agents: int, obs_dim: int, state_dim: int, device='cpu'):
        self.capacity = capacity
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim  # This will be updated dynamically based on actual states
        self.device = device

        # 使用deque提高效率
        self.buffer = deque(maxlen=capacity)

        # 动态检测实际的状态维度和观测维度
        self.actual_state_dim = None
        self.actual_obs_dims = None  # 用于处理异构观测维度
        
    def push(self, obs: Dict[str, np.ndarray],
             actions: Dict[str, int],
             rewards: Dict[str, float],
             next_obs: Dict[str, np.ndarray],
             dones: Dict[str, bool],
             global_state: np.ndarray,
             next_global_state: np.ndarray):
        """添加经验"""

        # 将字典转换为有序数组（按agent_id排序）
        agent_ids = sorted(obs.keys())

        # 检测并保存实际观测维度
        if self.actual_obs_dims is None:
            self.actual_obs_dims = [obs[agent_id].shape[0] for agent_id in agent_ids]

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

        # 确保global_state是一维数组
        global_state = np.asarray(global_state).flatten()
        next_global_state = np.asarray(next_global_state).flatten()

        # 动态检测并更新实际的状态维度
        if self.actual_state_dim is None:
            self.actual_state_dim = global_state.shape[0]

        # 对于动态状态维度（如Multiwalker），记录最大维度用于填充
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
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """采样批量数据"""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)

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
        # 检查是否是异构观测
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

            # 将多个张量堆叠成一个3D张量，但保持不同智能体的维度差异
            # 为了兼容现有算法，还是填充到相同维度，但记录原始维度信息
            max_obs_dim = max(obs.shape[-1] for obs in batch_obs[0])

            obs_batch = np.zeros((len(batch_obs), self.n_agents, max_obs_dim))
            next_obs_batch = np.zeros((len(batch_next_obs), self.n_agents, max_obs_dim))

            for i, (obs_list, next_obs_list) in enumerate(zip(batch_obs, batch_next_obs)):
                for j, (obs, next_obs) in enumerate(zip(obs_list, next_obs_list)):
                    obs_dim = obs.shape[0]
                    obs_batch[i, j, :obs_dim] = obs
                    next_obs_dim = next_obs.shape[0]
                    next_obs_batch[i, j, :next_obs_dim] = next_obs

            batch_obs_tensor = torch.FloatTensor(obs_batch).to(self.device)
            batch_next_obs_tensor = torch.FloatTensor(next_obs_batch).to(self.device)
        else:
            # 同构观测 - 直接转换
            batch_obs_tensor = torch.FloatTensor(np.array(batch_obs)).to(self.device)
            batch_next_obs_tensor = torch.FloatTensor(np.array(batch_next_obs)).to(self.device)

        # 处理不同维度的状态（如Multiwalker中的动态智能体数量）
        # 检查状态维度是否一致
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

        
        batch = {
            'obs': batch_obs_tensor,
            'actions': torch.LongTensor(np.array(batch_actions)).to(self.device),
            'rewards': torch.FloatTensor(np.array(batch_rewards)).to(self.device),
            'next_obs': batch_next_obs_tensor,
            'dones': torch.BoolTensor(np.array(batch_dones)).to(self.device),
            'global_state': global_state_tensor,
            'next_global_state': next_global_state_tensor
        }

        return batch
    
    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()