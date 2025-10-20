"""
QMIX 网络模型
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class AgentNetwork(nn.Module):
    """个体Q网络"""

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
            Q值，shape为(batch_size, action_dim)或(action_dim,)
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
            q_values = self.net(obs)

            # 恢复原始维度
            if len(original_shape) == 1:
                q_values = q_values.squeeze(0)

            return q_values
        except Exception as e:
            logger.error(f"前向传播时发生错误: {str(e)}")
            raise

    def act(self, obs, epsilon=0.0):
        """
        选择动作 (epsilon-greedy)

        Args:
            obs: 观测值
            epsilon: 探索率，范围[0, 1]

        Returns:
            动作索引
        """
        if not isinstance(epsilon, (int, float)):
            raise TypeError(f"epsilon必须是数值类型，当前类型为: {type(epsilon)}")

        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon必须在[0, 1]范围内，当前值为: {epsilon}")

        try:
            with torch.no_grad():
                q_values = self.forward(obs)

                # 确定batch_size
                if q_values.dim() == 1:
                    batch_size = 1
                    q_values = q_values.unsqueeze(0)
                else:
                    batch_size = q_values.shape[0]

                # epsilon-greedy策略
                if epsilon > 0 and torch.rand(1).item() < epsilon:
                    # 随机探索
                    action = torch.randint(0, self.action_dim, (batch_size,), 
                                         device=q_values.device)
                else:
                    # 贪婪选择
                    action = q_values.argmax(dim=-1)

                # 恢复原始维度
                if batch_size == 1:
                    action = action.squeeze(0)

                return action

        except Exception as e:
            logger.error(f"选择动作时发生错误: {str(e)}")
            raise


class MixingNetwork(nn.Module):
    """QMIX混合网络，保证单调性"""

    def __init__(self, n_agents: int, state_dim: int, mixing_hidden_dim: int = 32):
        super().__init__()

        # 参数验证和类型转换
        if not isinstance(n_agents, (int, np.integer)) or n_agents <= 0:
            raise ValueError(f"n_agents必须是正整数，当前值为: {n_agents}")
        if not isinstance(state_dim, (int, np.integer)) or state_dim <= 0:
            raise ValueError(f"state_dim必须是正整数，当前值为: {state_dim}")
        if not isinstance(mixing_hidden_dim, (int, np.integer)) or mixing_hidden_dim <= 0:
            raise ValueError(f"mixing_hidden_dim必须是正整数，当前值为: {mixing_hidden_dim}")
        
        # 转换为Python int类型
        n_agents = int(n_agents)
        state_dim = int(state_dim)
        mixing_hidden_dim = int(mixing_hidden_dim)

        self.n_agents = n_agents
        self.state_dim = state_dim
        self.mixing_hidden_dim = mixing_hidden_dim

        # 超网络：生成混合网络的权重
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, mixing_hidden_dim),
            nn.ReLU(),
            nn.Linear(mixing_hidden_dim, n_agents * mixing_hidden_dim)
        )

        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, mixing_hidden_dim),
            nn.ReLU(), 
            nn.Linear(mixing_hidden_dim, mixing_hidden_dim)
        )

        # 偏置网络
        self.hyper_b1 = nn.Sequential(
            nn.Linear(state_dim, mixing_hidden_dim),
            nn.ReLU(),
            nn.Linear(mixing_hidden_dim, mixing_hidden_dim)
        )

        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, mixing_hidden_dim),
            nn.ReLU(),
            nn.Linear(mixing_hidden_dim, 1)
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

    def forward(self, agent_qs, states):
        """
        前向传播

        Args:
            agent_qs: (batch_size, n_agents) 个体Q值
            states: (batch_size, state_dim) 全局状态

        Returns:
            q_tot: (batch_size,) 团队Q值
        """
        # 输入验证
        if agent_qs is None or states is None:
            raise ValueError("输入不能为None")

        if not isinstance(agent_qs, torch.Tensor) or not isinstance(states, torch.Tensor):
            raise TypeError("输入必须是torch.Tensor类型")

        # 维度检查
        if agent_qs.dim() != 2:
            raise ValueError(f"agent_qs维度必须是2，当前维度为: {agent_qs.dim()}")
        if states.dim() != 2:
            raise ValueError(f"states维度必须是2，当前维度为: {states.dim()}")

        batch_size = agent_qs.size(0)

        # 形状检查
        if agent_qs.size(1) != self.n_agents:
            raise ValueError(f"agent_qs的智能体数量不匹配，期望{self.n_agents}，实际{agent_qs.size(1)}")
        if states.size(0) != batch_size:
            raise ValueError(f"batch_size不匹配，agent_qs: {batch_size}, states: {states.size(0)}")
        if states.size(1) != self.state_dim:
            raise ValueError(f"state_dim不匹配，期望{self.state_dim}，实际{states.size(1)}")

        # 数值有效性检查
        if torch.isnan(agent_qs).any() or torch.isnan(states).any():
            raise ValueError("输入包含NaN")
        if torch.isinf(agent_qs).any() or torch.isinf(states).any():
            raise ValueError("输入包含Inf")

        try:
            # 生成权重（使用绝对值保证非负，确保单调性）
            w1 = torch.abs(self.hyper_w1(states))  # (batch_size, n_agents * mixing_hidden_dim)
            b1 = self.hyper_b1(states)              # (batch_size, mixing_hidden_dim)

            # 重塑权重矩阵
            w1 = w1.view(batch_size, self.n_agents, self.mixing_hidden_dim)

            # 第一层：加权组合
            agent_qs_expanded = agent_qs.unsqueeze(2)  # (batch_size, n_agents, 1)
            hidden = torch.sum(agent_qs_expanded * w1, dim=1) + b1  # (batch_size, mixing_hidden_dim)
            hidden = F.elu(hidden)

            # 第二层
            w2 = torch.abs(self.hyper_w2(states))  # (batch_size, mixing_hidden_dim)
            b2 = self.hyper_b2(states)              # (batch_size, 1)

            # 输出团队Q值
            q_tot = torch.sum(hidden * w2, dim=1, keepdim=True) + b2  # (batch_size, 1)
            q_tot = q_tot.squeeze(-1)  # (batch_size,)

            # 输出验证
            if torch.isnan(q_tot).any():
                logger.warning("混合网络输出包含NaN")
            if torch.isinf(q_tot).any():
                logger.warning("混合网络输出包含Inf")

            return q_tot

        except Exception as e:
            logger.error(f"混合网络前向传播时发生错误: {str(e)}")
            raise


class QMIXNetworks:
    """QMIX网络集合"""

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
        if 'mixing_hidden_dim' not in config['model']:
            raise KeyError("config['model']缺少必需的键: mixing_hidden_dim")

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
        mixing_hidden_dim = config['model']['mixing_hidden_dim']

        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim必须是正数，当前值为: {hidden_dim}")
        if mixing_hidden_dim <= 0:
            raise ValueError(f"mixing_hidden_dim必须是正数，当前值为: {mixing_hidden_dim}")

        try:
            # 创建个体网络
            if hasattr(self, 'heterogeneous_obs') and self.heterogeneous_obs:
                # 异构观测维度 - 为每个智能体创建对应维度的网络
                self.agent_networks = nn.ModuleList([
                    AgentNetwork(
                        self.obs_dims[i],  # 使用每个智能体的观测维度
                        self.action_dim,
                        hidden_dim
                    ).to(self.device)
                    for i in range(self.n_agents)
                ])
            else:
                # 同构观测维度
                self.agent_networks = nn.ModuleList([
                    AgentNetwork(
                        self.obs_dim,
                        self.action_dim,
                        hidden_dim
                    ).to(self.device)
                    for _ in range(self.n_agents)
                ])

            # 创建混合网络
            self.mixing_network = MixingNetwork(
                self.n_agents,
                self.state_dim,
                mixing_hidden_dim
            ).to(self.device)

            # 目标网络
            if hasattr(self, 'heterogeneous_obs') and self.heterogeneous_obs:
                # 异构观测维度
                self.target_agent_networks = nn.ModuleList([
                    AgentNetwork(
                        self.obs_dims[i],  # 使用每个智能体的观测维度
                        self.action_dim,
                        hidden_dim
                    ).to(self.device)
                    for i in range(self.n_agents)
                ])
            else:
                # 同构观测维度
                self.target_agent_networks = nn.ModuleList([
                    AgentNetwork(
                        self.obs_dim,
                        self.action_dim,
                        hidden_dim
                    ).to(self.device)
                    for _ in range(self.n_agents)
                ])

            self.target_mixing_network = MixingNetwork(
                self.n_agents,
                self.state_dim,
                mixing_hidden_dim
            ).to(self.device)

            # 初始化目标网络
            self.hard_update_target_networks()

            logger.info(f"成功创建QMIX网络: {self.n_agents}个智能体，"
                       f"观测维度{self.obs_dim}，动作维度{self.action_dim}，"
                       f"状态维度{self.state_dim}，设备{self.device}")

        except Exception as e:
            logger.error(f"创建QMIX网络时发生错误: {str(e)}")
            raise

    def hard_update_target_networks(self):
        """硬更新目标网络"""
        try:
            for agent, target_agent in zip(self.agent_networks, self.target_agent_networks):
                target_agent.load_state_dict(agent.state_dict())
            self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())
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
            # 更新个体网络
            for agent, target_agent in zip(self.agent_networks, self.target_agent_networks):
                for target_param, param in zip(target_agent.parameters(), agent.parameters()):
                    if target_param.data is None or param.data is None:
                        logger.warning("参数数据为None，跳过更新")
                        continue
                    target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

            # 更新混合网络
            for target_param, param in zip(self.target_mixing_network.parameters(),
                                          self.mixing_network.parameters()):
                if target_param.data is None or param.data is None:
                    logger.warning("参数数据为None，跳过更新")
                    continue
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

            logger.debug(f"目标网络软更新完成，tau={tau}")

        except Exception as e:
            logger.error(f"软更新目标网络时发生错误: {str(e)}")
            raise

    def get_all_parameters(self):
        """
        获取所有需要优化的参数

        Returns:
            参数列表
        """
        try:
            params = []

            # 收集个体网络参数
            for agent_net in self.agent_networks:
                params.extend(list(agent_net.parameters()))

            # 收集混合网络参数
            params.extend(list(self.mixing_network.parameters()))

            # 验证参数
            if len(params) == 0:
                logger.warning("没有可优化的参数")

            # 检查参数有效性
            for i, param in enumerate(params):
                if param is None:
                    logger.warning(f"参数{i}为None")
                elif not param.requires_grad:
                    logger.warning(f"参数{i}不需要梯度")

            logger.debug(f"收集到{len(params)}个参数用于优化")
            return params

        except Exception as e:
            logger.error(f"获取参数时发生错误: {str(e)}")
            raise