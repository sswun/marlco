"""
VDN 网络模型 - Value Decomposition Networks
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class AgentNetwork(nn.Module):
    """个体Q网络 - DRQN (Deep Recurrent Q Network)"""

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
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

        # GRU层用于处理序列数据
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0.0)

    def forward(self, obs, hidden_state=None):
        """
        前向传播

        Args:
            obs: 观测值，shape为(batch_size, obs_dim)或(obs_dim,)
            hidden_state: GRU隐藏状态，shape为(1, batch_size, hidden_dim)

        Returns:
            q_values: Q值，shape为(batch_size, action_dim)或(action_dim,)
            new_hidden_state: 新的隐藏状态
        """
        if obs is None:
            raise ValueError("观测值不能为None")

        if not isinstance(obs, torch.Tensor):
            raise TypeError(f"观测值必须是torch.Tensor类型，当前类型为: {type(obs)}")

        # 处理维度
        original_shape = obs.shape
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            batch_size = 1
        elif obs.dim() == 2:
            batch_size = obs.size(0)
        else:
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
            # 前馈层
            x = F.relu(self.fc1(obs))
            x = F.relu(self.fc2(x))

            # 处理GRU输入 - 需要序列维度
            if hidden_state is None:
                # 如果没有隐藏状态，创建初始隐藏状态
                hidden_state = torch.zeros(1, batch_size, self.hidden_dim,
                                         device=obs.device, dtype=obs.dtype)

            # GRU需要序列输入，将特征维度视为序列长度1
            x_gru = x.unsqueeze(1)  # (batch_size, 1, hidden_dim)

            # GRU前向传播
            x_gru, new_hidden_state = self.gru(x_gru, hidden_state)
            x = x_gru.squeeze(1)  # (batch_size, hidden_dim)

            # 输出层
            q_values = self.fc3(x)

            # 恢复原始维度
            if len(original_shape) == 1:
                q_values = q_values.squeeze(0)

            return q_values, new_hidden_state

        except Exception as e:
            logger.error(f"前向传播时发生错误: {str(e)}")
            raise

    def act(self, obs, hidden_state=None, epsilon=0.0):
        """
        选择动作 (epsilon-greedy)

        Args:
            obs: 观测值
            hidden_state: GRU隐藏状态
            epsilon: 探索率，范围[0, 1]

        Returns:
            action: 动作索引
            new_hidden_state: 新的隐藏状态
        """
        if not isinstance(epsilon, (int, float)):
            raise TypeError(f"epsilon必须是数值类型，当前类型为: {type(epsilon)}")

        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon必须在[0, 1]范围内，当前值为: {epsilon}")

        try:
            with torch.no_grad():
                q_values, new_hidden_state = self.forward(obs, hidden_state)

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

                return action, new_hidden_state

        except Exception as e:
            logger.error(f"选择动作时发生错误: {str(e)}")
            raise


class VDNNetworks:
    """VDN网络集合"""

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

            # 初始化目标网络
            self.hard_update_target_networks()

            logger.info(f"成功创建VDN网络: {self.n_agents}个智能体，"
                       f"观测维度{self.obs_dim}，动作维度{self.action_dim}，"
                       f"状态维度{self.state_dim}，设备{self.device}")

        except Exception as e:
            logger.error(f"创建VDN网络时发生错误: {str(e)}")
            raise

    def hard_update_target_networks(self):
        """硬更新目标网络"""
        try:
            for agent, target_agent in zip(self.agent_networks, self.target_agent_networks):
                target_agent.load_state_dict(agent.state_dict())
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

    def q_values(self, observations, hidden_states=None, use_target_network=False):
        """
        计算所有智能体的Q值

        Args:
            observations: (batch_size, n_agents, obs_dim) 观测值
            hidden_states: (n_agents, 1, batch_size, hidden_dim) 隐藏状态
            use_target_network: 是否使用目标网络

        Returns:
            q_values: (batch_size, n_agents, action_dim) Q值
            new_hidden_states: (n_agents, 1, batch_size, hidden_dim) 新的隐藏状态
        """
        if observations is None:
            raise ValueError("观测值不能为None")

        if not isinstance(observations, torch.Tensor):
            raise TypeError("观测值必须是torch.Tensor类型")

        # 维度检查
        if observations.dim() != 3:
            raise ValueError(f"observations维度必须是3，当前维度为: {observations.dim()}")

        batch_size = observations.size(0)

        # 形状检查
        if observations.size(1) != self.n_agents:
            raise ValueError(f"智能体数量不匹配，期望{self.n_agents}，实际{observations.size(1)}")
        if observations.size(2) != self.obs_dim:
            raise ValueError(f"观测维度不匹配，期望{self.obs_dim}，实际{observations.size(2)}")

        # 初始化隐藏状态
        if hidden_states is None:
            hidden_states = [None] * self.n_agents
        elif len(hidden_states) != self.n_agents:
            raise ValueError(f"隐藏状态数量不匹配，期望{self.n_agents}，实际{len(hidden_states)}")

        try:
            networks = self.target_agent_networks if use_target_network else self.agent_networks

            q_values_list = []
            new_hidden_states = []

            # 计算每个智能体的Q值
            for i in range(self.n_agents):
                obs_i = observations[:, i, :]  # (batch_size, obs_dim)
                hidden_i = hidden_states[i] if i < len(hidden_states) else None

                q_i, new_hidden_i = networks[i](obs_i, hidden_i)
                q_values_list.append(q_i)  # (batch_size, action_dim)
                new_hidden_states.append(new_hidden_i)  # (1, batch_size, hidden_dim)

            # 堆叠所有智能体的Q值
            q_values = torch.stack(q_values_list, dim=1)  # (batch_size, n_agents, action_dim)

            return q_values, new_hidden_states

        except Exception as e:
            logger.error(f"计算Q值时发生错误: {str(e)}")
            raise

    def q_total(self, q_values, actions):
        """
        计算VDN的总Q值（简单求和）

        Args:
            q_values: (batch_size, n_agents, action_dim) 个体Q值
            actions: (batch_size, n_agents) 选择的动作

        Returns:
            q_total: (batch_size,) 总Q值
        """
        if q_values is None or actions is None:
            raise ValueError("输入不能为None")

        if not isinstance(q_values, torch.Tensor) or not isinstance(actions, torch.Tensor):
            raise TypeError("输入必须是torch.Tensor类型")

        # 维度检查
        if q_values.dim() != 3:
            raise ValueError(f"q_values维度必须是3，当前维度为: {q_values.dim()}")
        if actions.dim() != 2:
            raise ValueError(f"actions维度必须是2，当前维度为: {actions.dim()}")

        batch_size = q_values.size(0)

        # 形状检查
        if q_values.size(1) != self.n_agents:
            raise ValueError(f"智能体数量不匹配，期望{self.n_agents}，实际{q_values.size(1)}")
        if actions.size(0) != batch_size:
            raise ValueError(f"batch_size不匹配，q_values: {batch_size}, actions: {actions.size(0)}")
        if actions.size(1) != self.n_agents:
            raise ValueError(f"智能体数量不匹配，期望{self.n_agents}，实际{actions.size(1)}")

        try:
            # 为每个智能体选择对应动作的Q值
            agent_q_values = []
            for i in range(self.n_agents):
                q_i = q_values[:, i, :]  # (batch_size, action_dim)
                a_i = actions[:, i]      # (batch_size,)
                q_selected = q_i.gather(1, a_i.unsqueeze(1)).squeeze(1)  # (batch_size,)
                agent_q_values.append(q_selected)

            # VDN核心：简单求和
            q_total = torch.stack(agent_q_values, dim=1).sum(dim=1)  # (batch_size,)

            return q_total

        except Exception as e:
            logger.error(f"计算总Q值时发生错误: {str(e)}")
            raise

    def save_model(self, filepath):
        """保存模型"""
        try:
            model_state = {
                'agent_networks': [net.state_dict() for net in self.agent_networks],
                'target_agent_networks': [net.state_dict() for net in self.target_agent_networks],
                'config': {
                    'n_agents': self.n_agents,
                    'obs_dim': self.obs_dim,
                    'action_dim': self.action_dim,
                    'state_dim': self.state_dim,
                    'heterogeneous_obs': getattr(self, 'heterogeneous_obs', False)
                }
            }

            if hasattr(self, 'obs_dims'):
                model_state['config']['obs_dims'] = self.obs_dims

            torch.save(model_state, filepath)
            logger.info(f"模型已保存到: {filepath}")

        except Exception as e:
            logger.error(f"保存模型时发生错误: {str(e)}")
            raise

    def load_model(self, filepath):
        """加载模型"""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)

            # 验证配置
            config = checkpoint['config']
            if config['n_agents'] != self.n_agents:
                raise ValueError(f"智能体数量不匹配: 文件{config['n_agents']} vs 当前{self.n_agents}")
            if config['obs_dim'] != self.obs_dim:
                raise ValueError(f"观测维度不匹配: 文件{config['obs_dim']} vs 当前{self.obs_dim}")
            if config['action_dim'] != self.action_dim:
                raise ValueError(f"动作维度不匹配: 文件{config['action_dim']} vs 当前{self.action_dim}")

            # 加载网络参数
            for i, (net_state, target_net_state) in enumerate(
                zip(checkpoint['agent_networks'], checkpoint['target_agent_networks'])):
                self.agent_networks[i].load_state_dict(net_state)
                self.target_agent_networks[i].load_state_dict(target_net_state)

            logger.info(f"模型已从{filepath}加载")

        except Exception as e:
            logger.error(f"加载模型时发生错误: {str(e)}")
            raise