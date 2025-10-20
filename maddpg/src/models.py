"""
MADDPG 网络模型
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ActorNetwork(nn.Module):
    """Actor网络 - 输出确定性动作"""

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
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, obs):
        """
        前向传播

        Args:
            obs: 观测值，shape为(batch_size, obs_dim)或(obs_dim,)

        Returns:
            动作概率，shape为(batch_size, action_dim)或(action_dim,)
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
            # 输出动作概率（softmax）
            action_logits = self.net(obs)
            action_probs = F.softmax(action_logits, dim=-1)

            # 恢复原始维度
            if len(original_shape) == 1:
                action_probs = action_probs.squeeze(0)

            return action_probs
        except Exception as e:
            logger.error(f"前向传播时发生错误: {str(e)}")
            raise

    def act(self, obs, noise_scale=0.0):
        """
        选择动作（从概率分布中采样）

        Args:
            obs: 观测值
            noise_scale: 噪声尺度（用于探索）

        Returns:
            动作索引
        """
        try:
            with torch.no_grad():
                action_probs = self.forward(obs)

                # 确定batch_size
                if action_probs.dim() == 1:
                    batch_size = 1
                    action_probs = action_probs.unsqueeze(0)
                else:
                    batch_size = action_probs.shape[0]

                # 添加探索噪声
                if noise_scale > 0:
                    noise = torch.randn_like(action_probs) * noise_scale
                    action_probs = action_probs + noise
                    action_probs = F.softmax(action_probs, dim=-1)

                # 从概率分布中采样
                action = torch.multinomial(action_probs, num_samples=1).squeeze(-1)

                # 恢复原始维度
                if batch_size == 1:
                    action = action.squeeze(0)

                return action

        except Exception as e:
            logger.error(f"选择动作时发生错误: {str(e)}")
            raise


class CriticNetwork(nn.Module):
    """
    Critic网络 - 集中式Q函数
    输入: 所有智能体的观测和动作
    输出: 单个智能体的Q值
    """

    def __init__(self, total_obs_dim: int, total_action_dim: int, hidden_dim: int = 64):
        super().__init__()

        # 参数验证和类型转换
        if not isinstance(total_obs_dim, (int, np.integer)) or total_obs_dim <= 0:
            raise ValueError(f"total_obs_dim必须是正整数，当前值为: {total_obs_dim}")
        if not isinstance(total_action_dim, (int, np.integer)) or total_action_dim <= 0:
            raise ValueError(f"total_action_dim必须是正整数，当前值为: {total_action_dim}")
        if not isinstance(hidden_dim, (int, np.integer)) or hidden_dim <= 0:
            raise ValueError(f"hidden_dim必须是正整数，当前值为: {hidden_dim}")
        
        # 转换为Python int类型
        total_obs_dim = int(total_obs_dim)
        total_action_dim = int(total_action_dim)
        hidden_dim = int(hidden_dim)

        self.total_obs_dim = total_obs_dim
        self.total_action_dim = total_action_dim
        self.hidden_dim = hidden_dim

        # 构建网络: 输入为所有智能体的观测和动作
        input_dim = total_obs_dim + total_action_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # 输出单个Q值
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

    def forward(self, obs, actions):
        """
        前向传播

        Args:
            obs: 所有智能体的观测，shape为(batch_size, total_obs_dim)
            actions: 所有智能体的动作，shape为(batch_size, total_action_dim)

        Returns:
            Q值，shape为(batch_size, 1)或(batch_size,)
        """
        # 输入验证
        if obs is None or actions is None:
            raise ValueError("输入不能为None")

        if not isinstance(obs, torch.Tensor) or not isinstance(actions, torch.Tensor):
            raise TypeError("输入必须是torch.Tensor类型")

        # 维度检查
        if obs.dim() != 2:
            raise ValueError(f"obs维度必须是2，当前维度为: {obs.dim()}")
        if actions.dim() != 2:
            raise ValueError(f"actions维度必须是2，当前维度为: {actions.dim()}")

        batch_size = obs.size(0)

        # 形状检查
        if obs.size(1) != self.total_obs_dim:
            raise ValueError(f"obs特征维度不匹配，期望{self.total_obs_dim}，实际{obs.size(1)}")
        if actions.size(1) != self.total_action_dim:
            raise ValueError(f"actions特征维度不匹配，期望{self.total_action_dim}，实际{actions.size(1)}")
        if actions.size(0) != batch_size:
            raise ValueError(f"batch_size不匹配，obs: {batch_size}, actions: {actions.size(0)}")

        # 数值有效性检查
        if torch.isnan(obs).any() or torch.isnan(actions).any():
            raise ValueError("输入包含NaN")
        if torch.isinf(obs).any() or torch.isinf(actions).any():
            raise ValueError("输入包含Inf")

        try:
            # 拼接观测和动作
            critic_input = torch.cat([obs, actions], dim=1)

            # 计算Q值
            q_value = self.net(critic_input)

            # 输出验证
            if torch.isnan(q_value).any():
                logger.warning("Critic网络输出包含NaN")
            if torch.isinf(q_value).any():
                logger.warning("Critic网络输出包含Inf")

            return q_value

        except Exception as e:
            logger.error(f"Critic前向传播时发生错误: {str(e)}")
            raise


class MADDPGNetworks:
    """MADDPG网络集合"""

    def __init__(self, env_info: Dict[str, Any], config: Dict[str, Any], device='cpu'):
        # 参数验证
        if not isinstance(env_info, dict):
            raise TypeError(f"env_info必须是字典类型，当前类型为: {type(env_info)}")
        if not isinstance(config, dict):
            raise TypeError(f"config必须是字典类型，当前类型为: {type(config)}")

        # 检查必需的键
        required_env_keys = ['n_agents', 'obs_dims', 'act_dims']
        for key in required_env_keys:
            if key not in env_info:
                raise KeyError(f"env_info缺少必需的键: {key}")

        required_config_keys = ['model']
        for key in required_config_keys:
            if key not in config:
                raise KeyError(f"config缺少必需的键: {key}")

        if 'actor_hidden_dim' not in config['model']:
            raise KeyError("config['model']缺少必需的键: actor_hidden_dim")
        if 'critic_hidden_dim' not in config['model']:
            raise KeyError("config['model']缺少必需的键: critic_hidden_dim")

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
            # 异构观测维度
            self.obs_dims = env_info['obs_dims']
            self.obs_dim = max(env_info['obs_dims'])
            self.heterogeneous_obs = True

        # 验证act_dims
        if not isinstance(env_info['act_dims'], (list, tuple)):
            raise TypeError("act_dims必须是列表或元组类型")
        if len(env_info['act_dims']) == 0:
            raise ValueError("act_dims不能为空")

        self.action_dim = env_info['act_dims'][0]
        
        # 转换为Python int类型
        self.obs_dim = int(self.obs_dim)
        self.action_dim = int(self.action_dim)

        # 计算总的观测和动作维度（用于Critic）
        if self.heterogeneous_obs:
            self.total_obs_dim = sum(self.obs_dims)
        else:
            self.total_obs_dim = self.obs_dim * self.n_agents
        self.total_action_dim = self.action_dim * self.n_agents

        # 验证维度值
        if self.obs_dim <= 0:
            raise ValueError(f"obs_dim必须是正数，当前值为: {self.obs_dim}")
        if self.action_dim <= 0:
            raise ValueError(f"action_dim必须是正数，当前值为: {self.action_dim}")

        # 设备验证
        if isinstance(device, str):
            if device not in ['cpu', 'cuda'] and not device.startswith('cuda:'):
                raise ValueError(f"不支持的device: {device}")
            if device.startswith('cuda') and not torch.cuda.is_available():
                logger.warning("CUDA不可用，将使用CPU")
                device = 'cpu'

        self.device = torch.device(device)

        # 提取配置参数
        actor_hidden_dim = config['model']['actor_hidden_dim']
        critic_hidden_dim = config['model']['critic_hidden_dim']

        if actor_hidden_dim <= 0:
            raise ValueError(f"actor_hidden_dim必须是正数，当前值为: {actor_hidden_dim}")
        if critic_hidden_dim <= 0:
            raise ValueError(f"critic_hidden_dim必须是正数，当前值为: {critic_hidden_dim}")

        try:
            # 为每个智能体创建Actor网络
            if hasattr(self, 'heterogeneous_obs') and self.heterogeneous_obs:
                # 异构观测维度
                self.actor_networks = nn.ModuleList([
                    ActorNetwork(
                        self.obs_dims[i],
                        self.action_dim,
                        actor_hidden_dim
                    ).to(self.device)
                    for i in range(self.n_agents)
                ])
            else:
                # 同构观测维度
                self.actor_networks = nn.ModuleList([
                    ActorNetwork(
                        self.obs_dim,
                        self.action_dim,
                        actor_hidden_dim
                    ).to(self.device)
                    for _ in range(self.n_agents)
                ])

            # 为每个智能体创建Critic网络（集中式）
            self.critic_networks = nn.ModuleList([
                CriticNetwork(
                    self.total_obs_dim,
                    self.total_action_dim,
                    critic_hidden_dim
                ).to(self.device)
                for _ in range(self.n_agents)
            ])

            # 创建目标网络
            if hasattr(self, 'heterogeneous_obs') and self.heterogeneous_obs:
                # 异构观测维度
                self.target_actor_networks = nn.ModuleList([
                    ActorNetwork(
                        self.obs_dims[i],
                        self.action_dim,
                        actor_hidden_dim
                    ).to(self.device)
                    for i in range(self.n_agents)
                ])
            else:
                # 同构观测维度
                self.target_actor_networks = nn.ModuleList([
                    ActorNetwork(
                        self.obs_dim,
                        self.action_dim,
                        actor_hidden_dim
                    ).to(self.device)
                    for _ in range(self.n_agents)
                ])

            self.target_critic_networks = nn.ModuleList([
                CriticNetwork(
                    self.total_obs_dim,
                    self.total_action_dim,
                    critic_hidden_dim
                ).to(self.device)
                for _ in range(self.n_agents)
            ])

            # 初始化目标网络
            self.hard_update_target_networks()

            logger.info(f"成功创建MADDPG网络: {self.n_agents}个智能体，"
                       f"观测维度{self.obs_dim}，动作维度{self.action_dim}，"
                       f"总观测维度{self.total_obs_dim}，总动作维度{self.total_action_dim}，"
                       f"设备{self.device}")

        except Exception as e:
            logger.error(f"创建MADDPG网络时发生错误: {str(e)}")
            raise

    def hard_update_target_networks(self):
        """硬更新目标网络"""
        try:
            for actor, target_actor in zip(self.actor_networks, self.target_actor_networks):
                target_actor.load_state_dict(actor.state_dict())
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
            # 更新Actor网络
            for actor, target_actor in zip(self.actor_networks, self.target_actor_networks):
                for target_param, param in zip(target_actor.parameters(), actor.parameters()):
                    if target_param.data is None or param.data is None:
                        logger.warning("参数数据为None，跳过更新")
                        continue
                    target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

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

    def get_actor_parameters(self, agent_id: int):
        """获取指定智能体的Actor参数"""
        return list(self.actor_networks[agent_id].parameters())

    def get_critic_parameters(self, agent_id: int):
        """获取指定智能体的Critic参数"""
        return list(self.critic_networks[agent_id].parameters())
