"""
COMA 算法核心实现
Counterfactual Multi-Agent Policy Gradients
"""
import torch
import torch.nn.functional as F
import torch.optim
from typing import Dict, Any, Tuple
import numpy as np
import logging
from .models import COMANetworks

logger = logging.getLogger(__name__)


class COMA:
    """COMA算法实现"""

    def __init__(self, networks: COMANetworks, config: Dict[str, Any], device='cpu'):
        self.networks = networks
        self.config = config
        self.device = device

        # 算法参数
        self.gamma = config['algorithm']['gamma']
        self.tau = config['algorithm']['tau']
        self.target_update_interval = config['algorithm']['target_update_interval']
        self.max_grad_norm = config['algorithm']['max_grad_norm']
        self.lambda_param = config['algorithm'].get('lambda', 0.8)  # TD(λ)参数

        # 优化器
        self.actor_optimizer = torch.optim.Adam(
            self.networks.get_actor_parameters(),
            lr=config['algorithm']['learning_rate']
        )
        self.critic_optimizer = torch.optim.Adam(
            self.networks.get_critic_parameters(),
            lr=config['algorithm']['learning_rate']
        )

        self.update_count = 0

    def select_actions(self, obs: Dict[str, torch.Tensor], epsilon: float = 0.0,
                      avail_actions: Dict[str, list] = None) -> Dict[str, int]:
        """选择动作 (epsilon-greedy)

        Args:
            obs: 观测字典
            epsilon: 探索率
            avail_actions: 可用动作字典 (agent_id -> list of available action indices)
        """
        actions = {}

        # 获取排序的agent_ids
        agent_ids = sorted(obs.keys())

        for i, agent_id in enumerate(agent_ids):
            # 将numpy数组转换为tensor
            agent_obs = torch.FloatTensor(obs[agent_id]).to(self.device)
            if agent_obs.dim() == 1:
                agent_obs = agent_obs.unsqueeze(0)

            # 对于异构观测，需要处理维度不匹配问题
            if hasattr(self.networks, 'heterogeneous_obs') and self.networks.heterogeneous_obs:
                actual_obs_dim = self.networks.obs_dims[i]
                current_dim = agent_obs.shape[-1]

                if current_dim < actual_obs_dim:
                    # 如果当前维度小于所需维度，进行填充
                    padding = torch.zeros(agent_obs.shape[0], actual_obs_dim - current_dim, device=self.device)
                    agent_obs = torch.cat([agent_obs, padding], dim=-1)
                elif current_dim > actual_obs_dim:
                    # 如果当前维度大于所需维度，进行裁剪
                    agent_obs = agent_obs[:, :actual_obs_dim]
            elif hasattr(self.networks, 'obs_dims') and len(self.networks.obs_dims) > len(set(self.networks.obs_dims)):
                # 检查是否为异构观测但没有标记
                current_dim = agent_obs.shape[-1]
                expected_dim = self.networks.obs_dim

                if current_dim != expected_dim:
                    if current_dim < expected_dim:
                        # 填充
                        padding = torch.zeros(agent_obs.shape[0], expected_dim - current_dim, device=self.device)
                        agent_obs = torch.cat([agent_obs, padding], dim=-1)
                    else:
                        # 裁剪
                        agent_obs = agent_obs[:, :expected_dim]

            with torch.no_grad():
                # 获取动作概率分布
                actor_net = self.networks.actor_networks[i]
                action_probs = actor_net.get_action_probs(agent_obs)

                # 如果提供了可用动作，则只从可用动作中选择
                if avail_actions is not None and agent_id in avail_actions and avail_actions[agent_id] is not None:
                    avail = avail_actions[agent_id]

                    if torch.rand(1).item() < epsilon:
                        # 随机探索 - 只从可用动作中随机选择
                        action = avail[torch.randint(0, len(avail), (1,)).item()]
                    else:
                        # 根据策略采样 - 只从可用动作中采样
                        avail_tensor = torch.tensor(avail, device=self.device)
                        # 确保action_probs是2D张量
                        if action_probs.dim() == 1:
                            action_probs = action_probs.unsqueeze(0)
                        avail_probs = action_probs[0, avail_tensor]
                        avail_probs = avail_probs / avail_probs.sum()  # 重新归一化
                        action_idx = torch.multinomial(avail_probs, 1).item()
                        action = avail[action_idx]
                else:
                    # 没有可用动作约束的情况
                    if torch.rand(1).item() < epsilon:
                        # 随机探索
                        action = torch.randint(0, self.networks.action_dim, (1,)).item()
                    else:
                        # 根据策略采样
                        # 确保action_probs是正确的维度
                        if action_probs.dim() == 1:
                            action_probs = action_probs.unsqueeze(0)
                        action = torch.multinomial(action_probs, 1).item()

            actions[agent_id] = action

        return actions

    def compute_critic_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算Critic损失 - TD(λ)算法"""
        obs = batch['obs']           # (batch_size, n_agents, obs_dim)
        actions = batch['actions']   # (batch_size, n_agents)
        rewards = batch['rewards']   # (batch_size, n_agents)
        next_obs = batch['next_obs'] # (batch_size, n_agents, obs_dim)
        dones = batch['dones']       # (batch_size, n_agents)
        states = batch['global_state']        # (batch_size, state_dim)
        next_states = batch['next_global_state'] # (batch_size, state_dim)

        batch_size = obs.size(0)
        critic_loss = 0

        # 对每个智能体的Critic计算损失
        for agent_id in range(self.networks.n_agents):
            # 计算当前Q值
            current_q_values = self.networks.critic_networks[agent_id](
                states, obs, actions, agent_id
            )  # (batch_size, action_dim)

            # 获取实际执行的动作的Q值
            agent_actions = actions[:, agent_id].long()
            current_q_taken = current_q_values.gather(1, agent_actions.unsqueeze(1)).squeeze(1)

            # 计算TD(λ)目标
            with torch.no_grad():
                # 下一个状态的最大Q值
                next_q_values = self.networks.target_critic_networks[agent_id](
                    next_states, next_obs, actions, agent_id
                )  # (batch_size, action_dim)

                # 团队奖励（所有智能体奖励之和）
                team_reward = rewards.sum(dim=1)  # (batch_size,)

                # 使用任一智能体的done标志
                done_mask = dones[:, 0].float()  # (batch_size,)

                # TD(λ)目标：简化版本，使用标准的TD目标
                target_q = team_reward + self.gamma * next_q_values.max(dim=1)[0] * (1 - done_mask)

            # 计算MSE损失
            agent_critic_loss = F.mse_loss(current_q_taken, target_q)
            critic_loss += agent_critic_loss

        return critic_loss / self.networks.n_agents

    def compute_actor_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算Actor损失 - 基于反事实优势函数"""
        obs = batch['obs']           # (batch_size, n_agents, obs_dim)
        actions = batch['actions']   # (batch_size, n_agents)
        states = batch['global_state']        # (batch_size, state_dim)

        batch_size = obs.size(0)
        actor_loss = 0

        # 对每个智能体计算策略梯度损失
        for agent_id in range(self.networks.n_agents):
            # 获取当前策略
            actor_net = self.networks.actor_networks[agent_id]
            agent_obs = obs[:, agent_id, :]

            # 对于异构观测，需要裁剪到该智能体的实际观测维度
            if hasattr(self.networks, 'heterogeneous_obs') and self.networks.heterogeneous_obs:
                actual_obs_dim = self.networks.obs_dims[agent_id]
                agent_obs = agent_obs[:, :actual_obs_dim]

            # 获取动作概率分布
            action_probs = actor_net.get_action_probs(agent_obs)  # (batch_size, action_dim)

            # 获取实际执行的动作的概率
            agent_actions = actions[:, agent_id].long()
            taken_action_probs = action_probs.gather(1, agent_actions.unsqueeze(1)).squeeze(1) + 1e-8

            # 计算反事实优势函数
            with torch.no_grad():
                # 1. 计算当前联合动作的Q值
                current_q_values = self.networks.critic_networks[agent_id](
                    states, obs, actions, agent_id
                )  # (batch_size, action_dim)
                current_q_taken = current_q_values.gather(1, agent_actions.unsqueeze(1)).squeeze(1)

                # 2. 计算反事实基线：对于当前智能体的所有可能动作，计算期望Q值
                # 注意：这里需要为每个可能的动作重新构建输入
                baseline = 0
                for action in range(self.networks.action_dim):
                    # 创建修改后的actions，将agent_id的动作改为action
                    modified_actions = actions.clone()
                    modified_actions[:, agent_id] = action

                    # 计算这个动作的Q值
                    q_for_action = self.networks.critic_networks[agent_id](
                        states, obs, modified_actions, agent_id
                    )  # (batch_size, action_dim)
                    action_tensor = torch.full((batch_size, 1), action,
                                              device=self.device, dtype=torch.long)
                    q_value = q_for_action.gather(1, action_tensor).squeeze(1)

                    # 加上该动作的概率
                    action_prob_for_action = action_probs[:, action]
                    baseline += action_prob_for_action * q_value

                # 3. 计算优势函数
                advantage = current_q_taken - baseline

            # 4. 计算策略梯度损失（使用负号，因为我们要最大化优势）
            # 使用log概率 * 优势
            log_probs = torch.log(taken_action_probs)
            agent_actor_loss = -(log_probs * advantage.detach()).mean()

            actor_loss += agent_actor_loss

        return actor_loss / self.networks.n_agents

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """更新网络"""
        # 计算Critic损失并更新
        critic_loss = self.compute_critic_loss(batch)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.networks.get_critic_parameters(),
            self.max_grad_norm
        )
        self.critic_optimizer.step()

        # 计算Actor损失并更新
        actor_loss = self.compute_actor_loss(batch)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.networks.get_actor_parameters(),
            self.max_grad_norm
        )
        self.actor_optimizer.step()

        # 更新目标网络
        self.update_count += 1
        if self.update_count % self.target_update_interval == 0:
            self.networks.hard_update_target_networks()
        else:
            self.networks.soft_update_target_networks(self.tau)

        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'update_count': self.update_count
        }

    def save(self, path: str):
        """保存模型"""
        state_dict = {
            'actor_networks': [net.state_dict() for net in self.networks.actor_networks],
            'critic_networks': [net.state_dict() for net in self.networks.critic_networks],
            'target_critic_networks': [net.state_dict() for net in self.networks.target_critic_networks],
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'update_count': self.update_count,
            'config': self.config
        }
        torch.save(state_dict, path)

    def load(self, path: str):
        """加载模型"""
        state_dict = torch.load(path, map_location=self.device)

        for i, actor_net in enumerate(self.networks.actor_networks):
            actor_net.load_state_dict(state_dict['actor_networks'][i])

        for i, critic_net in enumerate(self.networks.critic_networks):
            critic_net.load_state_dict(state_dict['critic_networks'][i])

        for i, target_critic_net in enumerate(self.networks.target_critic_networks):
            target_critic_net.load_state_dict(state_dict['target_critic_networks'][i])

        self.actor_optimizer.load_state_dict(state_dict['actor_optimizer'])
        self.critic_optimizer.load_state_dict(state_dict['critic_optimizer'])
        self.update_count = state_dict['update_count']