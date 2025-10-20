"""
QMIX 算法核心实现
"""
import torch
import torch.nn.functional as F
import torch.optim
from typing import Dict, Any, Tuple
from .models import QMIXNetworks


class QMIX:
    """QMIX算法实现"""
    
    def __init__(self, networks: QMIXNetworks, config: Dict[str, Any], device='cpu'):
        self.networks = networks
        self.config = config
        self.device = device
        
        # 算法参数
        self.gamma = config['algorithm']['gamma']
        self.tau = config['algorithm']['tau']
        self.target_update_interval = config['algorithm']['target_update_interval']
        self.max_grad_norm = config['algorithm']['max_grad_norm']
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            self.networks.get_all_parameters(),
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
        """计算QMIX损失"""
        obs = batch['obs']           # (batch_size, n_agents, obs_dim)
        actions = batch['actions']   # (batch_size, n_agents)
        rewards = batch['rewards']   # (batch_size, n_agents)
        next_obs = batch['next_obs'] # (batch_size, n_agents, obs_dim)
        dones = batch['dones']       # (batch_size, n_agents)
        states = batch['global_state']        # (batch_size, state_dim)
        next_states = batch['next_global_state'] # (batch_size, state_dim)
        
        batch_size = obs.size(0)
        
        # 1. 计算当前Q值
        current_agent_qs = []
        for i in range(self.networks.n_agents):
            agent_obs = obs[:, i, :]  # (batch_size, obs_dim)

            # 对于异构观测，需要裁剪到该智能体的实际观测维度
            if hasattr(self.networks, 'heterogeneous_obs') and self.networks.heterogeneous_obs:
                actual_obs_dim = self.networks.obs_dims[i]
                agent_obs = agent_obs[:, :actual_obs_dim]  # 裁剪到实际维度

            q_values = self.networks.agent_networks[i](agent_obs)  # (batch_size, action_dim)
            agent_actions = actions[:, i].long().unsqueeze(1)  # (batch_size, 1)
            q_taken = q_values.gather(1, agent_actions).squeeze(1)  # (batch_size,)
            current_agent_qs.append(q_taken)
        
        current_agent_qs = torch.stack(current_agent_qs, dim=1)  # (batch_size, n_agents)
        current_q_tot = self.networks.mixing_network(current_agent_qs, states)  # (batch_size,)
        
        # 2. 计算目标Q值
        with torch.no_grad():
            next_agent_qs = []
            for i in range(self.networks.n_agents):
                next_agent_obs = next_obs[:, i, :]  # (batch_size, obs_dim)

                # 对于异构观测，需要裁剪到该智能体的实际观测维度
                if hasattr(self.networks, 'heterogeneous_obs') and self.networks.heterogeneous_obs:
                    actual_obs_dim = self.networks.obs_dims[i]
                    next_agent_obs = next_agent_obs[:, :actual_obs_dim]  # 裁剪到实际维度

                next_q_values = self.networks.target_agent_networks[i](next_agent_obs)  # (batch_size, action_dim)
                
                # 如果batch中包含可用动作信息，需要mask不可用动作
                # 注意：这里假设所有环境都提供可用动作，或者都不提供
                # 对于不提供的环境，直接取max即可
                if 'avail_actions' in batch:
                    avail_actions_batch = batch['avail_actions']  # (batch_size, n_agents, action_dim)
                    avail_mask = avail_actions_batch[:, i, :]  # (batch_size, action_dim)
                    # 将不可用动作的Q值设为-inf
                    next_q_values = next_q_values.masked_fill(avail_mask == 0, float('-inf'))
                
                next_q_max = next_q_values.max(dim=1)[0]  # (batch_size,)
                next_agent_qs.append(next_q_max)
            
            next_agent_qs = torch.stack(next_agent_qs, dim=1)  # (batch_size, n_agents)
            next_q_tot = self.networks.target_mixing_network(next_agent_qs, next_states)  # (batch_size,)
            
            # 团队奖励（所有智能体奖励之和）
            team_reward = rewards.sum(dim=1)  # (batch_size,)
            
            # 使用任一智能体的done标志（假设所有智能体同时结束）
            done_mask = dones[:, 0].float()  # (batch_size,)
            
            # TD目标
            target_q = team_reward + self.gamma * next_q_tot * (1 - done_mask)
        
        # 3. 计算MSE损失
        loss = F.mse_loss(current_q_tot, target_q)
        
        return loss
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """更新网络"""
        # 计算损失
        loss = self.compute_loss(batch)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        total_norm = torch.nn.utils.clip_grad_norm_(
            self.networks.get_all_parameters(),
            self.max_grad_norm
        )
        
        self.optimizer.step()
        
        # 更新目标网络
        self.update_count += 1
        if self.update_count % self.target_update_interval == 0:
            self.networks.hard_update_target_networks()
        else:
            self.networks.soft_update_target_networks(self.tau)
        
        return {
            'loss': loss.item(),
            'grad_norm': total_norm.item(),
            'update_count': self.update_count
        }
    
    def save(self, path: str):
        """保存模型"""
        state_dict = {
            'agent_networks': [net.state_dict() for net in self.networks.agent_networks],
            'mixing_network': self.networks.mixing_network.state_dict(),
            'target_agent_networks': [net.state_dict() for net in self.networks.target_agent_networks],
            'target_mixing_network': self.networks.target_mixing_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'update_count': self.update_count,
            'config': self.config
        }
        torch.save(state_dict, path)
    
    def load(self, path: str):
        """加载模型"""
        state_dict = torch.load(path, map_location=self.device)
        
        for i, agent_net in enumerate(self.networks.agent_networks):
            agent_net.load_state_dict(state_dict['agent_networks'][i])
        
        self.networks.mixing_network.load_state_dict(state_dict['mixing_network'])
        
        for i, target_agent_net in enumerate(self.networks.target_agent_networks):
            target_agent_net.load_state_dict(state_dict['target_agent_networks'][i])
        
        self.networks.target_mixing_network.load_state_dict(state_dict['target_mixing_network'])
        
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.update_count = state_dict['update_count']