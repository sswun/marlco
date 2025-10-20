"""
IQL 算法核心实现
"""
import torch
import torch.nn.functional as F
import torch.optim
from typing import Dict, Any, Tuple
from .models import IQLNetworks


class IQL:
    """IQL算法实现 - Independent Q-Learning"""

    def __init__(self, networks: IQLNetworks, config: Dict[str, Any], device='cpu'):
        self.networks = networks
        self.config = config
        self.device = device

        # 算法参数
        self.gamma = config['algorithm']['gamma']
        self.tau = config['algorithm']['tau']
        self.target_update_interval = config['algorithm']['target_update_interval']
        self.max_grad_norm = config['algorithm']['max_grad_norm']

        # 优化器 - 为每个智能体创建独立的优化器
        self.optimizers = []
        for agent_net in self.networks.agent_networks:
            optimizer = torch.optim.Adam(
                agent_net.parameters(),
                lr=config['algorithm']['learning_rate']
            )
            self.optimizers.append(optimizer)

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
            agent_obs = obs[agent_id].unsqueeze(0) if obs[agent_id].dim() == 1 else obs[agent_id]

            # 对于异构观测，需要裁剪到该智能体的实际观测维度
            if hasattr(self.networks, 'heterogeneous_obs') and self.networks.heterogeneous_obs:
                actual_obs_dim = self.networks.obs_dims[i]
                agent_obs = agent_obs[:, :actual_obs_dim]  # 裁剪到实际维度

            with torch.no_grad():
                q_values = self.networks.agent_networks[i](agent_obs)

                # 如果提供了可用动作，则只从可用动作中选择
                if avail_actions is not None and agent_id in avail_actions and avail_actions[agent_id] is not None:
                    avail = avail_actions[agent_id]

                    if torch.rand(1).item() < epsilon:
                        # 随机探索 - 只从可用动作中随机选择
                        action = avail[torch.randint(0, len(avail), (1,)).item()]
                    else:
                        # 贪婪选择 - mask不可用动作后选择最大Q值
                        q_values_masked = q_values.clone()
                        # 将不可用动作的Q值设为-inf
                        mask = torch.ones_like(q_values_masked) * float('-inf')
                        mask[0, avail] = 0
                        q_values_masked = q_values_masked + mask
                        action = q_values_masked.argmax(dim=-1).item()
                else:
                    # 没有可用动作约束的情况（兼容其他环境）
                    if torch.rand(1).item() < epsilon:
                        # 随机探索
                        action = torch.randint(0, q_values.size(-1), (1,)).item()
                    else:
                        # 贪婪选择
                        action = q_values.argmax(dim=-1).item()

            actions[agent_id] = action

        return actions

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算IQL损失 - 为每个智能体独立计算损失"""
        obs = batch['obs']           # (batch_size, n_agents, obs_dim)
        actions = batch['actions']   # (batch_size, n_agents)
        rewards = batch['rewards']   # (batch_size, n_agents)
        next_obs = batch['next_obs'] # (batch_size, n_agents, obs_dim)
        dones = batch['dones']       # (batch_size, n_agents)

        batch_size = obs.size(0)

        total_loss = 0.0

        # 为每个智能体独立计算损失
        for i in range(self.networks.n_agents):
            # 获取当前智能体的数据
            agent_obs = obs[:, i, :]  # (batch_size, obs_dim)
            agent_actions = actions[:, i].long().unsqueeze(1)  # (batch_size, 1)
            agent_rewards = rewards[:, i]  # (batch_size,)
            agent_next_obs = next_obs[:, i, :]  # (batch_size, obs_dim)
            agent_dones = dones[:, i]  # (batch_size,)

            # 对于异构观测，需要裁剪到该智能体的实际观测维度
            if hasattr(self.networks, 'heterogeneous_obs') and self.networks.heterogeneous_obs:
                actual_obs_dim = self.networks.obs_dims[i]
                agent_obs = agent_obs[:, :actual_obs_dim]  # 裁剪到实际维度
                agent_next_obs = agent_next_obs[:, :actual_obs_dim]  # 裁剪到实际维度

            # 1. 计算当前Q值
            current_q_values = self.networks.agent_networks[i](agent_obs)  # (batch_size, action_dim)
            current_q = current_q_values.gather(1, agent_actions).squeeze(1)  # (batch_size,)

            # 2. 计算目标Q值
            with torch.no_grad():
                next_q_values = self.networks.target_agent_networks[i](agent_next_obs)  # (batch_size, action_dim)

                # 如果batch中包含可用动作信息，需要mask不可用动作
                # 注意：这里假设所有环境都提供可用动作，或者都不提供
                # 对于不提供的环境，直接取max即可
                if 'avail_actions' in batch:
                    avail_actions_batch = batch['avail_actions']  # (batch_size, n_agents, action_dim)
                    avail_mask = avail_actions_batch[:, i, :]  # (batch_size, action_dim)
                    # 将不可用动作的Q值设为-inf
                    next_q_values = next_q_values.masked_fill(avail_mask == 0, float('-inf'))

                next_q_max = next_q_values.max(dim=1)[0]  # (batch_size,)

                # TD目标
                target_q = agent_rewards + self.gamma * next_q_max * (1 - agent_dones.float())

            # 3. 计算MSE损失
            agent_loss = F.mse_loss(current_q, target_q)
            total_loss += agent_loss

        # 平均损失
        avg_loss = total_loss / self.networks.n_agents

        return avg_loss

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """更新网络 - 为每个智能体独立更新"""
        # 为每个智能体独立计算损失和更新
        agent_losses = []
        agent_grad_norms = []

        obs = batch['obs']           # (batch_size, n_agents, obs_dim)
        actions = batch['actions']   # (batch_size, n_agents)
        rewards = batch['rewards']   # (batch_size, n_agents)
        next_obs = batch['next_obs'] # (batch_size, n_agents, obs_dim)
        dones = batch['dones']       # (batch_size, n_agents)

        batch_size = obs.size(0)

        # 为每个智能体独立更新
        for i in range(self.networks.n_agents):
            # 获取当前智能体的数据
            agent_obs = obs[:, i, :]  # (batch_size, obs_dim)
            agent_actions = actions[:, i].long().unsqueeze(1)  # (batch_size, 1)
            agent_rewards = rewards[:, i]  # (batch_size,)
            agent_next_obs = next_obs[:, i, :]  # (batch_size, obs_dim)
            agent_dones = dones[:, i]  # (batch_size,)

            # 对于异构观测，需要裁剪到该智能体的实际观测维度
            if hasattr(self.networks, 'heterogeneous_obs') and self.networks.heterogeneous_obs:
                actual_obs_dim = self.networks.obs_dims[i]
                agent_obs = agent_obs[:, :actual_obs_dim]  # 裁剪到实际维度
                agent_next_obs = agent_next_obs[:, :actual_obs_dim]  # 裁剪到实际维度

            # 1. 计算当前Q值
            current_q_values = self.networks.agent_networks[i](agent_obs)  # (batch_size, action_dim)
            current_q = current_q_values.gather(1, agent_actions).squeeze(1)  # (batch_size,)

            # 2. 计算目标Q值
            with torch.no_grad():
                next_q_values = self.networks.target_agent_networks[i](agent_next_obs)  # (batch_size, action_dim)

                # 如果batch中包含可用动作信息，需要mask不可用动作
                if 'avail_actions' in batch:
                    avail_actions_batch = batch['avail_actions']  # (batch_size, n_agents, action_dim)
                    avail_mask = avail_actions_batch[:, i, :]  # (batch_size, action_dim)
                    # 将不可用动作的Q值设为-inf
                    next_q_values = next_q_values.masked_fill(avail_mask == 0, float('-inf'))

                next_q_max = next_q_values.max(dim=1)[0]  # (batch_size,)

                # TD目标
                target_q = agent_rewards + self.gamma * next_q_max * (1 - agent_dones.float())

            # 3. 计算MSE损失
            agent_loss = F.mse_loss(current_q, target_q)

            # 4. 反向传播和参数更新
            self.optimizers[i].zero_grad()
            agent_loss.backward()

            # 梯度裁剪
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.networks.agent_networks[i].parameters(),
                self.max_grad_norm
            )

            self.optimizers[i].step()

            agent_losses.append(agent_loss.item())
            agent_grad_norms.append(grad_norm.item())

        # 更新目标网络
        self.update_count += 1
        if self.update_count % self.target_update_interval == 0:
            self.networks.hard_update_target_networks()
        else:
            self.networks.soft_update_target_networks(self.tau)

        # 返回统计信息
        avg_loss = sum(agent_losses) / len(agent_losses)
        avg_grad_norm = sum(agent_grad_norms) / len(agent_grad_norms)

        return {
            'loss': avg_loss,
            'grad_norm': avg_grad_norm,
            'update_count': self.update_count,
            'agent_losses': agent_losses,
            'agent_grad_norms': agent_grad_norms
        }

    def save(self, path: str):
        """保存模型"""
        state_dict = {
            'agent_networks': [net.state_dict() for net in self.networks.agent_networks],
            'target_agent_networks': [net.state_dict() for net in self.networks.target_agent_networks],
            'optimizers': [opt.state_dict() for opt in self.optimizers],
            'update_count': self.update_count,
            'config': self.config
        }
        torch.save(state_dict, path)

    def load(self, path: str):
        """加载模型"""
        state_dict = torch.load(path, map_location=self.device)

        for i, agent_net in enumerate(self.networks.agent_networks):
            agent_net.load_state_dict(state_dict['agent_networks'][i])

        for i, target_agent_net in enumerate(self.networks.target_agent_networks):
            target_agent_net.load_state_dict(state_dict['target_agent_networks'][i])

        for i, optimizer in enumerate(self.optimizers):
            optimizer.load_state_dict(state_dict['optimizers'][i])

        self.update_count = state_dict['update_count']