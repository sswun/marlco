"""
MAPPO 网络模型
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def init_orthogonal(module, gain=1.0):
    """正交初始化"""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


class ActorNetwork(nn.Module):
    """Actor网络 - 输出动作概率分布"""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64, 
                 use_orthogonal_init: bool = True):
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

        # 初始化权重
        if use_orthogonal_init:
            self.apply(lambda m: init_orthogonal(m, gain=0.01))
        else:
            self._initialize_weights()

    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, obs):
        """
        前向传播

        Args:
            obs: 观测值，shape为(batch_size, obs_dim)或(obs_dim,)

        Returns:
            动作logits，shape为(batch_size, action_dim)或(action_dim,)
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
            x = F.relu(self.fc1(obs))
            x = F.relu(self.fc2(x))
            action_logits = self.fc3(x)

            # 恢复原始维度
            if len(original_shape) == 1:
                action_logits = action_logits.squeeze(0)

            return action_logits
        except Exception as e:
            logger.error(f"前向传播时发生错误: {str(e)}")
            raise

    def get_action_probs(self, obs):
        """获取动作概率"""
        logits = self.forward(obs)
        return F.softmax(logits, dim=-1)

    def evaluate_actions(self, obs, actions):
        """
        评估动作

        Args:
            obs: 观测
            actions: 动作索引

        Returns:
            action_log_probs: 动作对数概率
            dist_entropy: 策略熵
        """
        logits = self.forward(obs)
        
        # 使用log_softmax和softmax的数值稳定计算（不裁剪logits）
        log_probs = F.log_softmax(logits, dim=-1)
        probs = F.softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(1, actions.long().unsqueeze(-1)).squeeze(-1)
        
        # 计算熵（添加数值稳定性）
        dist_entropy = -(probs * log_probs).sum(dim=-1).mean()
        
        # 防止熵为nan
        if torch.isnan(dist_entropy):
            dist_entropy = torch.tensor(0.0, device=dist_entropy.device)
        
        return action_log_probs, dist_entropy


class CriticNetwork(nn.Module):
    """
    Critic网络 - 集中式价值函数
    输入: 全局状态（所有智能体的观测）
    输出: 状态价值V
    """

    def __init__(self, state_dim: int, hidden_dim: int = 64, use_orthogonal_init: bool = True):
        super().__init__()

        # 参数验证和类型转换
        if not isinstance(state_dim, (int, np.integer)) or state_dim <= 0:
            raise ValueError(f"state_dim必须是正整数，当前值为: {state_dim}")
        if not isinstance(hidden_dim, (int, np.integer)) or hidden_dim <= 0:
            raise ValueError(f"hidden_dim必须是正整数，当前值为: {hidden_dim}")
        
        # 转换为Python int类型
        state_dim = int(state_dim)
        hidden_dim = int(hidden_dim)

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        # 构建网络
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        # 初始化权重
        if use_orthogonal_init:
            self.apply(lambda m: init_orthogonal(m, gain=1.0))
        else:
            self._initialize_weights()

    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, state):
        """
        前向传播

        Args:
            state: 全局状态，shape为(batch_size, state_dim)

        Returns:
            value: 状态价值，shape为(batch_size, 1)
        """
        # 输入验证
        if state is None:
            raise ValueError("状态不能为None")

        if not isinstance(state, torch.Tensor):
            raise TypeError("状态必须是torch.Tensor类型")

        # 维度检查
        if state.dim() != 2:
            raise ValueError(f"state维度必须是2，当前维度为: {state.dim()}")

        # 形状检查
        if state.size(1) != self.state_dim:
            raise ValueError(f"state维度不匹配，期望{self.state_dim}，实际{state.size(1)}")

        # 数值有效性检查
        if torch.isnan(state).any():
            raise ValueError("状态包含NaN")
        if torch.isinf(state).any():
            raise ValueError("状态包含Inf")

        try:
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
            value = self.fc3(x)

            # 输出验证
            if torch.isnan(value).any():
                logger.warning("Critic网络输出包含NaN")
            if torch.isinf(value).any():
                logger.warning("Critic网络输出包含Inf")

            return value

        except Exception as e:
            logger.error(f"Critic前向传播时发生错误: {str(e)}")
            raise


class MAPPONetworks:
    """MAPPO网络集合"""

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

        # 计算全局状态维度（用于Critic）
        if self.heterogeneous_obs:
            self.state_dim = sum(self.obs_dims)
        else:
            self.state_dim = self.obs_dim * self.n_agents

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
        use_orthogonal_init = config['model'].get('use_orthogonal_init', True)

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
                        actor_hidden_dim,
                        use_orthogonal_init
                    ).to(self.device)
                    for i in range(self.n_agents)
                ])
            else:
                # 同构观测维度
                self.actor_networks = nn.ModuleList([
                    ActorNetwork(
                        self.obs_dim,
                        self.action_dim,
                        actor_hidden_dim,
                        use_orthogonal_init
                    ).to(self.device)
                    for _ in range(self.n_agents)
                ])

            # 创建集中式Critic网络（共享）
            self.critic_network = CriticNetwork(
                self.state_dim,
                critic_hidden_dim,
                use_orthogonal_init
            ).to(self.device)

            logger.info(f"成功创建MAPPO网络: {self.n_agents}个智能体，"
                       f"观测维度{self.obs_dim}，动作维度{self.action_dim}，"
                       f"全局状态维度{self.state_dim}，设备{self.device}")

        except Exception as e:
            logger.error(f"创建MAPPO网络时发生错误: {str(e)}")
            raise

    def get_actor_parameters(self, agent_id: int):
        """获取指定智能体的Actor参数"""
        return list(self.actor_networks[agent_id].parameters())

    def get_critic_parameters(self):
        """获取Critic参数"""
        return list(self.critic_network.parameters())

    def get_all_actor_parameters(self):
        """获取所有Actor参数"""
        params = []
        for actor_net in self.actor_networks:
            params.extend(list(actor_net.parameters()))
        return params
