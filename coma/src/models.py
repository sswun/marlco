"""
COMA 网络模型
实现基于反事实多智能体策略梯度的Actor-Critic架构
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ActorNetwork(nn.Module):
    """Actor网络：策略网络，输出动作概率分布"""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()

        # 参数验证和类型转换
        if not isinstance(obs_dim, (int, np.integer)) or obs_dim <= 0:
            raise ValueError(f"obs_dim必须是正整数，当前值为: {obs_dim}")
        if not isinstance(action_dim, (int, np.integer)) or action_dim <= 0:
            raise ValueError(f"action_dim必须是正整数，当前值为: {action_dim}")
        if not isinstance(hidden_dim, (int, np.integer)) or hidden_dim <= 0:
            raise ValueError(f"hidden_dim必须是正整数，当前值为: {hidden_dim}")

        # 转换为Python int类型
        obs_dim = int(obs_dim)
        action_dim = int(action_dim)
        hidden_dim = int(hidden_dim)

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # 构建网络
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)  # 小的初始化对于策略网络更重要
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, obs):
        """
        前向传播，输出动作logits

        Args:
            obs: 观测值，shape为(batch_size, obs_dim)或(obs_dim,)

        Returns:
            logits: 动作logits，shape为(batch_size, action_dim)或(action_dim,)
        """
        if obs is None:
            raise ValueError("观测值不能为None")

        if not isinstance(obs, torch.Tensor):
            raise TypeError(f"观测值必须是torch.Tensor类型，当前类型为: {type(obs)}")

        # 处理维度
        original_shape = obs.shape
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        elif obs.dim() != 2:
            raise ValueError(f"观测值维度必须是1或2，当前维度为: {obs.dim()}")

        # 检查特征维度
        if obs.shape[-1] != self.obs_dim:
            raise ValueError(f"观测值特征维度不匹配，期望{self.obs_dim}，实际{obs.shape[-1]}")

        # 检查数值有效性
        if torch.isnan(obs).any():
            raise ValueError("观测值包含NaN")
        if torch.isinf(obs).any():
            raise ValueError("观测值包含Inf")

        try:
            logits = self.net(obs)

            # 恢复原始维度
            if len(original_shape) == 1:
                logits = logits.squeeze(0)

            return logits
        except Exception as e:
            logger.error(f"Actor前向传播时发生错误: {str(e)}")
            raise

    def get_action_probs(self, obs, avail_actions=None):
        """
        获取动作概率分布

        Args:
            obs: 观测值
            avail_actions: 可用动作mask (batch_size, action_dim)

        Returns:
            action_probs: 动作概率分布
        """
        logits = self.forward(obs)

        if avail_actions is not None:
            # mask不可用动作
            mask = (avail_actions == 0)
            logits = logits.masked_fill(mask, float('-inf'))

        action_probs = F.softmax(logits, dim=-1)
        return action_probs


class CriticNetwork(nn.Module):
    """Critic网络：中心化评论家，计算反事实优势函数"""

    def __init__(self, obs_dim: int, action_dim: int, n_agents: int,
                 state_dim: int, hidden_dim: int = 64, obs_dims=None):
        super().__init__()

        # 参数验证和类型转换
        if not isinstance(obs_dim, (int, np.integer)) or obs_dim <= 0:
            raise ValueError(f"obs_dim必须是正整数，当前值为: {obs_dim}")
        if not isinstance(action_dim, (int, np.integer)) or action_dim <= 0:
            raise ValueError(f"action_dim必须是正整数，当前值为: {action_dim}")
        if not isinstance(n_agents, (int, np.integer)) or n_agents <= 0:
            raise ValueError(f"n_agents必须是正整数，当前值为: {n_agents}")
        if not isinstance(state_dim, (int, np.integer)) or state_dim <= 0:
            raise ValueError(f"state_dim必须是正整数，当前值为: {state_dim}")
        if not isinstance(hidden_dim, (int, np.integer)) or hidden_dim <= 0:
            raise ValueError(f"hidden_dim必须是正整数，当前值为: {hidden_dim}")

        # 转换为Python int类型
        obs_dim = int(obs_dim)
        action_dim = int(action_dim)
        n_agents = int(n_agents)
        state_dim = int(state_dim)
        hidden_dim = int(hidden_dim)

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.obs_dims = obs_dims  # 保存各智能体的观测维度

        # 输入维度：全局状态 + 所有智能体观测 + 其他智能体动作
        # 对于计算智能体a的优势，输入不包含a的动作，但输出a对所有动作的Q值
        if obs_dims is not None:
            # 异构观测维度 - 使用实际观测维度之和
            total_obs_dim = sum(obs_dims)
        else:
            # 同构观测维度
            total_obs_dim = obs_dim * n_agents

        input_dim = state_dim + total_obs_dim + action_dim * (n_agents - 1)

        # 构建网络
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)  # 输出指定智能体对所有动作的Q值
        )

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, state, all_obs, all_actions, agent_id):
        """
        前向传播，计算指定智能体对所有可能动作的Q值

        Args:
            state: 全局状态 (batch_size, state_dim)
            all_obs: 所有智能体观测 (batch_size, n_agents, obs_dim)
            all_actions: 所有智能体动作 (batch_size, n_agents)
            agent_id: 要计算的智能体ID (int)

        Returns:
            q_values: 指定智能体对所有动作的Q值 (batch_size, action_dim)
        """
        # 输入验证
        if state is None or all_obs is None or all_actions is None:
            raise ValueError("输入不能为None")

        batch_size = state.size(0)

        # 构建输入：状态 + 所有观测 + 除agent_id外的其他动作
        critic_inputs = []

        # 1. 全局状态
        critic_inputs.append(state)

        # 2. 所有智能体的观测（处理异构观测维度）
        if self.obs_dims is not None:
            # 异构观测维度 - 分别处理每个智能体的观测
            obs_list = []
            for i in range(self.n_agents):
                obs_list.append(all_obs[:, i, :self.obs_dims[i]])
            critic_inputs.append(torch.cat(obs_list, dim=-1))
        else:
            # 同构观测维度 - 直接展平
            critic_inputs.append(all_obs.view(batch_size, -1))

        # 3. 其他智能体的动作（one-hot编码并排除agent_id）
        for i in range(self.n_agents):
            if i != agent_id:
                actions_one_hot = F.one_hot(all_actions[:, i], num_classes=self.action_dim).float()
                critic_inputs.append(actions_one_hot)

        # 拼接所有输入
        critic_input = torch.cat(critic_inputs, dim=-1)

        # 前向传播
        q_values = self.net(critic_input)

        return q_values


class COMANetworks:
    """COMA网络集合，包含Actor和Critic网络"""

    def __init__(self, env_info: Dict[str, Any], config: Dict[str, Any], device='cpu'):
        # 参数验证
        if not isinstance(env_info, dict):
            raise TypeError(f"env_info必须是字典类型，当前类型为: {type(env_info)}")
        if not isinstance(config, dict):
            raise TypeError(f"config必须是字典类型，当前类型为: {type(config)}")

        # 检查必需的键
        required_env_keys = ['n_agents', 'obs_dims', 'act_dims', 'global_state_dim']
        for key in required_env_keys:
            if key not in env_info:
                raise KeyError(f"env_info缺少必需的键: {key}")

        required_config_keys = ['model']
        for key in required_config_keys:
            if key not in config:
                raise KeyError(f"config缺少必需的键: {key}")

        if 'hidden_dim' not in config['model']:
            raise KeyError("config['model']缺少必需的键: hidden_dim")

        # 提取参数
        self.n_agents = env_info['n_agents']

        # 验证和转换n_agents
        if not isinstance(self.n_agents, (int, np.integer)) or self.n_agents <= 0:
            raise ValueError(f"n_agents必须是正整数，当前值为: {self.n_agents}")
        self.n_agents = int(self.n_agents)

        # 验证obs_dims
        if not isinstance(env_info['obs_dims'], (list, tuple)):
            raise TypeError("obs_dims必须是列表或元组类型")
        if len(env_info['obs_dims']) == 0:
            raise ValueError("obs_dims不能为空")

        # 检查是否所有智能体观测维度相同
        if len(set(env_info['obs_dims'])) == 1:
            # 同构观测维度
            self.obs_dim = env_info['obs_dims'][0]
            self.heterogeneous_obs = False
        else:
            # 异构观测维度 - 使用最大观测维度并填充较小的观测
            self.obs_dims = env_info['obs_dims']  # 保存每个智能体的观测维度
            self.obs_dim = max(env_info['obs_dims'])  # 使用最大观测维度
            self.heterogeneous_obs = True

        # 验证act_dims
        if not isinstance(env_info['act_dims'], (list, tuple)):
            raise TypeError("act_dims必须是列表或元组类型")
        if len(env_info['act_dims']) == 0:
            raise ValueError("act_dims不能为空")

        self.action_dim = env_info['act_dims'][0]
        self.state_dim = env_info['global_state_dim']

        # 转换为Python int类型（可能是numpy类型）
        self.obs_dim = int(self.obs_dim)
        self.action_dim = int(self.action_dim)
        self.state_dim = int(self.state_dim)

        # 验证维度值
        if self.obs_dim <= 0:
            raise ValueError(f"obs_dim必须是正数，当前值为: {self.obs_dim}")
        if self.action_dim <= 0:
            raise ValueError(f"action_dim必须是正数，当前值为: {self.action_dim}")
        if self.state_dim <= 0:
            raise ValueError(f"state_dim必须是正数，当前值为: {self.state_dim}")

        # 设备验证
        if isinstance(device, str):
            if device not in ['cpu', 'cuda'] and not device.startswith('cuda:'):
                raise ValueError(f"不支持的device: {device}")
            if device.startswith('cuda') and not torch.cuda.is_available():
                logger.warning("CUDA不可用，将使用CPU")
                device = 'cpu'

        self.device = torch.device(device)

        # 提取配置参数
        hidden_dim = config['model']['hidden_dim']

        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim必须是正数，当前值为: {hidden_dim}")

        try:
            # 创建Actor网络（参数共享）
            if hasattr(self, 'heterogeneous_obs') and self.heterogeneous_obs:
                # 异构观测维度 - 为每个智能体创建对应维度的网络
                self.actor_networks = nn.ModuleList([
                    ActorNetwork(
                        self.obs_dims[i],  # 使用每个智能体的观测维度
                        self.action_dim,
                        hidden_dim
                    ).to(self.device)
                    for i in range(self.n_agents)
                ])
            else:
                # 同构观测维度 - 所有智能体共享同一个Actor网络
                self.shared_actor = ActorNetwork(
                    self.obs_dim,
                    self.action_dim,
                    hidden_dim
                ).to(self.device)
                # 为兼容性创建ModuleList
                self.actor_networks = nn.ModuleList([self.shared_actor for _ in range(self.n_agents)])

            # 创建Critic网络（每个智能体一个Critic）
            if hasattr(self, 'heterogeneous_obs') and self.heterogeneous_obs:
                self.critic_networks = nn.ModuleList([
                    CriticNetwork(
                        self.obs_dims[i],  # 使用每个智能体的观测维度
                        self.action_dim,
                        self.n_agents,
                        self.state_dim,
                        hidden_dim,
                        obs_dims=self.obs_dims  # 传递所有智能体的观测维度
                    ).to(self.device)
                    for i in range(self.n_agents)
                ])
            else:
                self.critic_networks = nn.ModuleList([
                    CriticNetwork(
                        self.obs_dim,
                        self.action_dim,
                        self.n_agents,
                        self.state_dim,
                        hidden_dim,
                        obs_dims=None  # 同构观测维度
                    ).to(self.device)
                    for _ in range(self.n_agents)
                ])

            # 目标Critic网络
            if hasattr(self, 'heterogeneous_obs') and self.heterogeneous_obs:
                self.target_critic_networks = nn.ModuleList([
                    CriticNetwork(
                        self.obs_dims[i],  # 使用每个智能体的观测维度
                        self.action_dim,
                        self.n_agents,
                        self.state_dim,
                        hidden_dim,
                        obs_dims=self.obs_dims  # 传递所有智能体的观测维度
                    ).to(self.device)
                    for i in range(self.n_agents)
                ])
            else:
                self.target_critic_networks = nn.ModuleList([
                    CriticNetwork(
                        self.obs_dim,
                        self.action_dim,
                        self.n_agents,
                        self.state_dim,
                        hidden_dim,
                        obs_dims=None  # 同构观测维度
                    ).to(self.device)
                    for _ in range(self.n_agents)
                ])

            # 初始化目标网络
            self.hard_update_target_networks()

            logger.info(f"成功创建COMA网络: {self.n_agents}个智能体，"
                       f"观测维度{self.obs_dim}，动作维度{self.action_dim}，"
                       f"状态维度{self.state_dim}，设备{self.device}")

        except Exception as e:
            logger.error(f"创建COMA网络时发生错误: {str(e)}")
            raise

    def hard_update_target_networks(self):
        """硬更新目标网络"""
        try:
            for critic, target_critic in zip(self.critic_networks, self.target_critic_networks):
                target_critic.load_state_dict(critic.state_dict())
            logger.debug("目标网络硬更新完成")
        except Exception as e:
            logger.error(f"硬更新目标网络时发生错误: {str(e)}")
            raise

    def soft_update_target_networks(self, tau=0.005):
        """
        软更新目标网络

        Args:
            tau: 软更新系数，范围(0, 1]
        """
        if not isinstance(tau, (int, float)):
            raise TypeError(f"tau必须是数值类型，当前类型为: {type(tau)}")

        if not 0.0 < tau <= 1.0:
            raise ValueError(f"tau必须在(0, 1]范围内，当前值为: {tau}")

        try:
            # 更新Critic网络
            for critic, target_critic in zip(self.critic_networks, self.target_critic_networks):
                for target_param, param in zip(target_critic.parameters(), critic.parameters()):
                    if target_param.data is None or param.data is None:
                        logger.warning("参数数据为None，跳过更新")
                        continue
                    target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

            logger.debug(f"目标网络软更新完成，tau={tau}")

        except Exception as e:
            logger.error(f"软更新目标网络时发生错误: {str(e)}")
            raise

    def get_actor_parameters(self):
        """获取Actor网络参数"""
        try:
            params = []
            for actor_net in self.actor_networks:
                params.extend(list(actor_net.parameters()))
            return params
        except Exception as e:
            logger.error(f"获取Actor参数时发生错误: {str(e)}")
            raise

    def get_critic_parameters(self):
        """获取Critic网络参数"""
        try:
            params = []
            for critic_net in self.critic_networks:
                params.extend(list(critic_net.parameters()))
            return params
        except Exception as e:
            logger.error(f"获取Critic参数时发生错误: {str(e)}")
            raise