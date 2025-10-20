"""
VDN 算法核心实现
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class VDN:
    """VDN算法实现"""

    def __init__(self, env_info: Dict[str, Any], config: Dict[str, Any], device='cpu'):
        """
        初始化VDN算法

        Args:
            env_info: 环境信息
            config: 配置参数
            device: 计算设备
        """
        # 参数验证
        if not isinstance(env_info, dict):
            raise TypeError(f"env_info必须是字典类型，当前类型为: {type(env_info)}")
        if not isinstance(config, dict):
            raise TypeError(f"config必须是字典类型，当前类型为: {type(config)}")

        # 检查必需的配置键
        required_keys = ['algorithm', 'model', 'training', 'exploration']
        for key in required_keys:
            if key not in config:
                raise KeyError(f"config缺少必需的键: {key}")

        self.env_info = env_info
        self.config = config
        self.device = device

        # 智能体ID
        self.agent_ids = env_info.get('agent_ids', [f'agent_{i}' for i in range(env_info['n_agents'])])

        # 检查是否有异构观测维度
        self.heterogeneous_obs = len(set(env_info['obs_dims'])) > 1

        # 算法参数
        self.gamma = config['algorithm'].get('gamma', 0.99)
        self.learning_rate = config['algorithm'].get('learning_rate', 0.001)
        self.tau = config['algorithm'].get('tau', 0.005)
        self.target_update_interval = config['algorithm'].get('target_update_interval', 50)
        self.max_grad_norm = config['algorithm'].get('max_grad_norm', 10.0)

        # 创建网络（延迟导入避免循环依赖）
        from .models import VDNNetworks
        self.networks = VDNNetworks(env_info, config, device)

        # 创建优化器
        self.optimizer = optim.Adam(
            self.networks.get_all_parameters(),
            lr=self.learning_rate
        )

        # 训练状态
        self.training_steps = 0
        self.episodes = 0

        logger.info(f"VDN算法初始化完成: gamma={self.gamma}, lr={self.learning_rate}")

    def select_actions(self, observations, hidden_states=None, epsilon=0.0):
        """
        选择动作

        Args:
            observations: (n_agents, obs_dim) 观测值
            hidden_states: (n_agents, 1, 1, hidden_dim) 隐藏状态
            epsilon: 探索率

        Returns:
            actions: (n_agents,) 选择的动作
            new_hidden_states: (n_agents, 1, 1, hidden_dim) 新的隐藏状态
        """
        if observations is None:
            raise ValueError("观测值不能为None")

        if not isinstance(observations, torch.Tensor):
            raise TypeError("观测值必须是torch.Tensor类型")

        # 确保观测值在正确的设备上
        observations = observations.to(self.device)

        # 维度检查和调整
        if observations.dim() == 1:
            observations = observations.unsqueeze(0)  # (1, obs_dim)

        if observations.dim() != 2:
            raise ValueError(f"观测值维度必须是2，当前维度为: {observations.dim()}")

        # 初始化隐藏状态
        if hidden_states is None:
            hidden_states = [None] * self.networks.n_agents

        try:
            actions = []
            new_hidden_states = []

            # 为每个智能体选择动作
            for i in range(self.networks.n_agents):
                obs_i = observations[i:i+1, :] if observations.size(0) == 1 else observations[i, :]  # (1, obs_dim) or (obs_dim,)
                hidden_i = hidden_states[i] if i < len(hidden_states) else None

                action_i, new_hidden_i = self.networks.agent_networks[i].act(obs_i, hidden_i, epsilon)
                actions.append(action_i.item() if isinstance(action_i, torch.Tensor) else action_i)
                new_hidden_states.append(new_hidden_i)

            actions = torch.tensor(actions, device=self.device, dtype=torch.long)
            return actions, new_hidden_states

        except Exception as e:
            logger.error(f"选择动作时发生错误: {str(e)}")
            raise

    def select_actions_heterogeneous(self, obs_list, hidden_states=None, epsilon=0.0):
        """
        为异构观测环境选择动作

        Args:
            obs_list: 每个智能体的观测列表 [obs_0, obs_1, ..., obs_n]
            hidden_states: 每个智能体的隐藏状态列表
            epsilon: 探索率

        Returns:
            actions: (n_agents,) 选择的动作
            new_hidden_states: (n_agents,) 新的隐藏状态列表
        """
        if obs_list is None:
            raise ValueError("观测值不能为None")

        if not isinstance(obs_list, list):
            raise TypeError("异构观测模式下，观测值必须是列表类型")

        if len(obs_list) != self.networks.n_agents:
            raise ValueError(f"观测列表长度({len(obs_list)})与智能体数量({self.networks.n_agents})不匹配")

        # 初始化隐藏状态
        if hidden_states is None:
            hidden_states = [None] * self.networks.n_agents
        elif not isinstance(hidden_states, list):
            hidden_states = [hidden_states] * self.networks.n_agents

        try:
            actions = []
            new_hidden_states = []

            # 为每个智能体选择动作
            for i in range(self.networks.n_agents):
                obs_i = obs_list[i].unsqueeze(0) if obs_list[i].dim() == 1 else obs_list[i]  # 确保 (1, obs_dim)
                obs_i = obs_i.to(self.device)
                hidden_i = hidden_states[i] if i < len(hidden_states) else None

                action_i, new_hidden_i = self.networks.agent_networks[i].act(obs_i, hidden_i, epsilon)
                actions.append(action_i.item() if isinstance(action_i, torch.Tensor) else action_i)
                new_hidden_states.append(new_hidden_i)

            actions = torch.tensor(actions, device=self.device, dtype=torch.long)
            return actions, new_hidden_states

        except Exception as e:
            logger.error(f"异构观测选择动作时发生错误: {str(e)}")
            raise

    def compute_loss(self, batch):
        """
        计算VDN损失

        Args:
            batch: 训练批次数据

        Returns:
            loss: 损失值
            metrics: 指标字典
        """
        if batch is None:
            raise ValueError("批次数据不能为None")

        try:
            # 解包批次数据
            (obs_batch, actions_batch, rewards_batch, next_obs_batch,
             done_batch, global_states_batch, next_global_states_batch) = batch

            # 确保张量在正确的设备上
            # 处理异构观测
            if isinstance(obs_batch, list):
                # 异构观测 - 保持列表格式，每个元素已经在正确的设备上
                pass  # 不需要转换
            else:
                # 同构观测 - 转换为张量
                if not isinstance(obs_batch, torch.Tensor):
                    obs_batch = torch.FloatTensor(obs_batch).to(self.device)
                elif obs_batch.device != self.device:
                    obs_batch = obs_batch.to(self.device)

            if not isinstance(actions_batch, torch.Tensor):
                actions_batch = torch.LongTensor(actions_batch).to(self.device)
            elif actions_batch.device != self.device:
                actions_batch = actions_batch.to(self.device)

            if not isinstance(rewards_batch, torch.Tensor):
                rewards_batch = torch.FloatTensor(rewards_batch).to(self.device)
            elif rewards_batch.device != self.device:
                rewards_batch = rewards_batch.to(self.device)

            # 处理异构next_obs
            if isinstance(next_obs_batch, list):
                # 异构观测 - 保持列表格式
                pass  # 不需要转换
            else:
                # 同构观测 - 转换为张量
                if not isinstance(next_obs_batch, torch.Tensor):
                    next_obs_batch = torch.FloatTensor(next_obs_batch).to(self.device)
                elif next_obs_batch.device != self.device:
                    next_obs_batch = next_obs_batch.to(self.device)

            if not isinstance(done_batch, torch.Tensor):
                done_batch = torch.FloatTensor(done_batch).to(self.device)
            elif done_batch.device != self.device:
                done_batch = done_batch.to(self.device)

            # VDN使用团队奖励，所以需要将每个智能体的奖励求和
            if rewards_batch.dim() == 2:
                rewards_batch = rewards_batch.sum(dim=1)  # [batch_size, n_agents] -> [batch_size]

            # 同样处理done标志
            if done_batch.dim() == 2:
                done_batch = done_batch.any(dim=1).float()  # [batch_size, n_agents] -> [batch_size]

            # 处理异构观测
            if isinstance(obs_batch, list):
                # 异构观测 - 分别计算每个智能体的Q值
                batch_size = obs_batch[0].size(0)

                # 1. 计算当前Q值
                current_q_values = []
                for i in range(self.networks.n_agents):
                    obs_i = obs_batch[i]  # (batch_size, obs_dim_i)
                    q_values_i, _ = self.networks.agent_networks[i].forward(obs_i)
                    current_q_values.append(q_values_i)

                # 将Q值堆叠成 (batch_size, n_agents, action_dim) 格式
                # 需要确保所有action_dim相同（通常是的）
                current_q_values = torch.stack(current_q_values, dim=1)  # (batch_size, n_agents, action_dim)
                current_q_total = self.networks.q_total(current_q_values, actions_batch)

                # 2. 计算目标Q值
                with torch.no_grad():
                    # 计算下一状态的Q值
                    next_q_values = []
                    for i in range(self.networks.n_agents):
                        next_obs_i = next_obs_batch[i]  # (batch_size, obs_dim_i)
                        next_q_values_i, _ = self.networks.target_agent_networks[i].forward(next_obs_i)
                        next_q_values.append(next_q_values_i)

                    next_q_values = torch.stack(next_q_values, dim=1)  # (batch_size, n_agents, action_dim)

                    # 为每个智能体选择最大Q值的动作
                    next_max_actions = next_q_values.argmax(dim=-1)  # (batch_size, n_agents)

                    # 计算下一状态的总Q值
                    next_q_total = self.networks.q_total(next_q_values, next_max_actions)
            else:
                # 同构观测 - 使用原始方法
                batch_size = obs_batch.size(0)

                # 1. 计算当前Q值
                current_q_values, _ = self.networks.q_values(obs_batch, use_target_network=False)
                current_q_total = self.networks.q_total(current_q_values, actions_batch)

                # 2. 计算目标Q值
                with torch.no_grad():
                    # 使用目标网络计算下一状态的最大Q值
                    next_q_values, _ = self.networks.q_values(next_obs_batch, use_target_network=True)

                    # 为每个智能体选择最大Q值的动作
                    next_max_actions = next_q_values.argmax(dim=-1)  # (batch_size, n_agents)

                    # 计算下一状态的总Q值
                    next_q_total = self.networks.q_total(next_q_values, next_max_actions)

            # VDN TD目标
            target_q_total = rewards_batch + self.gamma * (1.0 - done_batch) * next_q_total

            # 3. 计算损失
            loss = F.mse_loss(current_q_total, target_q_total)

            # 4. 计算指标
            with torch.no_grad():
                metrics = {
                    'loss': loss.item(),
                    'current_q_mean': current_q_total.mean().item(),
                    'target_q_mean': target_q_total.mean().item(),
                    'q_error': (current_q_total - target_q_total).abs().mean().item(),
                    'rewards_mean': rewards_batch.mean().item(),
                    'max_q_value': current_q_values.max().item(),
                    'min_q_value': current_q_values.min().item()
                }

            return loss, metrics

        except Exception as e:
            logger.error(f"计算损失时发生错误: {str(e)}")
            raise

    def train_step(self, batch):
        """
        执行一个训练步骤

        Args:
            batch: 训练批次数据

        Returns:
            metrics: 训练指标
        """
        if batch is None:
            raise ValueError("批次数据不能为None")

        try:
            # 计算损失
            loss, metrics = self.compute_loss(batch)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.networks.get_all_parameters(),
                    self.max_grad_norm
                )

            # 更新参数
            self.optimizer.step()

            # 更新训练步数
            self.training_steps += 1

            # 软更新目标网络
            if self.training_steps % self.target_update_interval == 0:
                self.networks.soft_update_target_networks(self.tau)
                metrics['target_network_updated'] = True
            else:
                metrics['target_network_updated'] = False

            # 添加训练信息
            metrics['training_steps'] = self.training_steps
            metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']

            return metrics

        except Exception as e:
            logger.error(f"训练步骤时发生错误: {str(e)}")
            raise

    def evaluate_episode(self, env, max_steps=None, render=False):
        """
        评估一个episode

        Args:
            env: 环境
            max_steps: 最大步数
            render: 是否渲染

        Returns:
            episode_reward: episode总奖励
            episode_length: episode长度
            metrics: 评估指标
        """
        if env is None:
            raise ValueError("环境不能为None")

        try:
            episode_reward = 0.0
            episode_length = 0

            # 重置环境
            observations, info = env.reset()
            if not isinstance(observations, dict):
                # 转换为字典格式
                observations = {self.agent_ids[i]: observations[i] for i in range(len(observations))}

            # 检查是否为异构观测
            heterogeneous_obs = len(set(observations[agent_id].shape[0] for agent_id in self.agent_ids)) > 1

            if heterogeneous_obs:
                # 异构观测维度 - 使用列表形式
                obs_list = [
                    torch.FloatTensor(observations[self.agent_ids[i]]).to(self.device)
                    for i in range(self.networks.n_agents)
                ]
            else:
                # 同构观测维度 - 可以堆叠
                obs_tensor = torch.stack([
                    torch.FloatTensor(observations[self.agent_ids[i]])
                    for i in range(self.networks.n_agents)
                ]).to(self.device)

            hidden_states = None

            # Episode循环
            done = False
            while not done:
                # 选择动作（无探索）
                if heterogeneous_obs:
                    actions, hidden_states = self.select_actions_heterogeneous(
                        obs_list, hidden_states, epsilon=0.0
                    )
                else:
                    actions, hidden_states = self.select_actions(
                        obs_tensor, hidden_states, epsilon=0.0
                    )

                # 转换动作格式
                action_dict = {self.agent_ids[i]: actions[i].item() for i in range(len(actions))}

                # 执行动作
                step_result = env.step(action_dict)
                if len(step_result) == 4:
                    next_observations, rewards, terminated, info = step_result
                    truncated = False
                else:
                    next_observations, rewards, terminated, truncated, info = step_result

                # 转换观测格式
                if not isinstance(next_observations, dict):
                    next_observations = {self.agent_ids[i]: next_observations[i] for i in range(len(next_observations))}

                # 累积奖励
                if isinstance(rewards, dict):
                    reward = sum(rewards.values())
                else:
                    reward = rewards

                episode_reward += reward
                episode_length += 1

                # 检查是否结束
                if isinstance(terminated, dict):
                    done = any(terminated.values()) or any(truncated.values()) if isinstance(truncated, dict) else any(terminated.values()) or truncated
                else:
                    done = terminated or truncated
                if isinstance(done, dict):
                    done = all(done.values())

                # 更新观测
                observations = next_observations
                if heterogeneous_obs:
                    # 异构观测维度 - 更新列表
                    obs_list = [
                        torch.FloatTensor(observations[self.agent_ids[i]]).to(self.device)
                        for i in range(self.networks.n_agents)
                    ]
                else:
                    # 同构观测维度 - 更新张量
                    obs_tensor = torch.stack([
                        torch.FloatTensor(observations[self.agent_ids[i]])
                        for i in range(self.networks.n_agents)
                    ]).to(self.device)

                # 检查最大步数
                if max_steps is not None and episode_length >= max_steps:
                    break

            # 评估指标
            metrics = {
                'episode_reward': episode_reward,
                'episode_length': episode_length,
                'average_reward': episode_reward / episode_length if episode_length > 0 else 0.0
            }

            return episode_reward, episode_length, metrics

        except Exception as e:
            logger.error(f"评估episode时发生错误: {str(e)}")
            raise

    def save_model(self, filepath):
        """
        保存模型

        Args:
            filepath: 保存路径
        """
        try:
            self.networks.save_model(filepath)

            # 保存算法状态
            algo_state = {
                'training_steps': self.training_steps,
                'episodes': self.episodes,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': self.config
            }

            # 保存算法状态到另一个文件
            import os
            algo_filepath = filepath.replace('.pth', '_algo.pth')
            torch.save(algo_state, algo_filepath)

            logger.info(f"算法状态已保存到: {algo_filepath}")

        except Exception as e:
            logger.error(f"保存模型时发生错误: {str(e)}")
            raise

    def load_model(self, filepath):
        """
        加载模型

        Args:
            filepath: 加载路径
        """
        try:
            self.networks.load_model(filepath)

            # 尝试加载算法状态
            import os
            algo_filepath = filepath.replace('.pth', '_algo.pth')
            if os.path.exists(algo_filepath):
                algo_state = torch.load(algo_filepath, map_location=self.device)
                self.training_steps = algo_state.get('training_steps', 0)
                self.episodes = algo_state.get('episodes', 0)
                self.optimizer.load_state_dict(algo_state['optimizer_state_dict'])
                logger.info(f"算法状态已从{algo_filepath}加载")
            else:
                logger.warning(f"算法状态文件不存在: {algo_filepath}")

        except Exception as e:
            logger.error(f"加载模型时发生错误: {str(e)}")
            raise

    def get_network_info(self):
        """
        获取网络信息

        Returns:
            info: 网络信息字典
        """
        try:
            # 计算网络参数数量
            total_params = sum(p.numel() for p in self.networks.get_all_parameters())

            info = {
                'n_agents': self.networks.n_agents,
                'obs_dim': self.networks.obs_dim,
                'action_dim': self.networks.action_dim,
                'state_dim': self.networks.state_dim,
                'total_parameters': total_params,
                'device': str(self.device),
                'training_steps': self.training_steps,
                'episodes': self.episodes,
                'gamma': self.gamma,
                'learning_rate': self.learning_rate,
                'tau': self.tau
            }

            # 添加异构观测信息
            if hasattr(self.networks, 'heterogeneous_obs') and self.networks.heterogeneous_obs:
                info['heterogeneous_obs'] = True
                info['obs_dims'] = self.networks.obs_dims
            else:
                info['heterogeneous_obs'] = False

            return info

        except Exception as e:
            logger.error(f"获取网络信息时发生错误: {str(e)}")
            raise

    def set_training_mode(self, training=True):
        """
        设置训练模式

        Args:
            training: 是否为训练模式
        """
        try:
            for network in self.networks.agent_networks:
                network.train(training)

            for network in self.networks.target_agent_networks:
                network.train(training)

            logger.debug(f"网络设置为{'训练' if training else '评估'}模式")

        except Exception as e:
            logger.error(f"设置训练模式时发生错误: {str(e)}")
            raise