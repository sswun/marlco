"""
MADDPG 算法核心实现
"""
import torch
import torch.nn.functional as F
import torch.optim
import numpy as np
from typing import Dict, Any, Tuple
from .models import MADDPGNetworks


class MADDPG:
    """MADDPG算法实现"""
    
    def __init__(self, networks: MADDPGNetworks, config: Dict[str, Any], device='cpu'):
        self.networks = networks
        self.config = config
        self.device = device
        self.n_agents = networks.n_agents
        
        # 算法参数
        self.gamma = config['algorithm']['gamma']
        self.tau = config['algorithm']['tau']
        self.max_grad_norm = config['algorithm']['max_grad_norm']
        
        # 为每个智能体创建优化器
        self.actor_optimizers = []
        self.critic_optimizers = []
        
        for i in range(self.n_agents):
            # Actor优化器
            actor_optimizer = torch.optim.Adam(
                self.networks.get_actor_parameters(i),
                lr=config['algorithm']['actor_lr']
            )
            self.actor_optimizers.append(actor_optimizer)
            
            # Critic优化器
            critic_optimizer = torch.optim.Adam(
                self.networks.get_critic_parameters(i),
                lr=config['algorithm']['critic_lr']
            )
            self.critic_optimizers.append(critic_optimizer)
        
        self.update_count = 0
    
    def select_actions(self, obs: Dict[str, torch.Tensor], noise_scale: float = 0.0, 
                      avail_actions: Dict[str, list] = None) -> Dict[str, int]:
        """
        选择动作
        
        Args:
            obs: 观测字典
            noise_scale: 噪声尺度（用于探索）
            avail_actions: 可用动作字典
        """
        actions = {}
        
        # 获取排序的agent_ids
        agent_ids = sorted(obs.keys())
        
        for i, agent_id in enumerate(agent_ids):
            agent_obs = obs[agent_id].unsqueeze(0) if obs[agent_id].dim() == 1 else obs[agent_id]

            # 对于异构观测，需要裁剪到该智能体的实际观测维度
            if hasattr(self.networks, 'heterogeneous_obs') and self.networks.heterogeneous_obs:
                actual_obs_dim = self.networks.obs_dims[i]
                agent_obs = agent_obs[:, :actual_obs_dim]

            with torch.no_grad():
                # 使用Actor网络选择动作
                action = self.networks.actor_networks[i].act(agent_obs, noise_scale)
                
                # 如果提供了可用动作，确保选择的动作在可用范围内
                if avail_actions is not None and agent_id in avail_actions and avail_actions[agent_id] is not None:
                    avail = avail_actions[agent_id]
                    # 如果选择的动作不可用，从可用动作中随机选择
                    if action.item() not in avail:
                        action = torch.tensor(avail[torch.randint(0, len(avail), (1,)).item()], 
                                            device=action.device)
                
                actions[agent_id] = action.item()
            
        return actions
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """计算MADDPG损失"""
        obs = batch['obs']           # (batch_size, n_agents, obs_dim)
        actions = batch['actions']   # (batch_size, n_agents)
        rewards = batch['rewards']   # (batch_size, n_agents)
        next_obs = batch['next_obs'] # (batch_size, n_agents, obs_dim)
        dones = batch['dones']       # (batch_size, n_agents)
        
        batch_size = obs.size(0)
        
        critic_losses = {}
        actor_losses = {}
        
        # 将观测和动作展平为集中式输入
        # 对于异构观测，需要裁剪到每个智能体的实际观测维度
        if hasattr(self.networks, 'heterogeneous_obs') and self.networks.heterogeneous_obs:
            # 异构观测：分别裁剪每个智能体的观测维度
            obs_list = []
            next_obs_list = []
            for i in range(self.n_agents):
                actual_obs_dim = self.networks.obs_dims[i]
                obs_list.append(obs[:, i, :actual_obs_dim])
                next_obs_list.append(next_obs[:, i, :actual_obs_dim])
            obs_flat = torch.cat(obs_list, dim=1)  # (batch_size, total_obs_dim)
            next_obs_flat = torch.cat(next_obs_list, dim=1)
        else:
            # 同构观测：直接展平
            obs_flat = obs.view(batch_size, -1)
            next_obs_flat = next_obs.view(batch_size, -1)
        
        # 将动作转换为one-hot编码
        # actions: (batch_size, n_agents) -> (batch_size, n_agents, action_dim)
        actions_onehot = F.one_hot(actions.long(), num_classes=self.networks.action_dim).float()
        # (batch_size, n_agents, action_dim) -> (batch_size, total_action_dim)
        actions_flat = actions_onehot.view(batch_size, -1)
        
        # 为每个智能体计算损失
        for i in range(self.n_agents):
            # ===== 更新Critic =====
            with torch.no_grad():
                # 使用目标Actor网络获取下一状态的动作
                next_actions_list = []
                for j in range(self.n_agents):
                    next_agent_obs = next_obs[:, j, :]
                    
                    # 对于异构观测，需要裁剪到该智能体的实际观测维度
                    if hasattr(self.networks, 'heterogeneous_obs') and self.networks.heterogeneous_obs:
                        actual_obs_dim = self.networks.obs_dims[j]
                        next_agent_obs = next_agent_obs[:, :actual_obs_dim]
                    
                    next_action_probs = self.networks.target_actor_networks[j](next_agent_obs)
                    next_actions_list.append(next_action_probs)
                
                # 拼接所有智能体的下一动作概率
                next_actions_concat = torch.cat(next_actions_list, dim=1)
                
                # 使用目标Critic网络计算目标Q值
                target_q = self.networks.target_critic_networks[i](next_obs_flat, next_actions_concat)
                
                # 计算TD目标
                # 使用当前智能体的奖励和done标志
                agent_reward = rewards[:, i].unsqueeze(1)  # (batch_size, 1)
                agent_done = dones[:, i].float().unsqueeze(1)  # (batch_size, 1)
                
                y = agent_reward + self.gamma * target_q * (1 - agent_done)
            
            # 当前Q值
            current_q = self.networks.critic_networks[i](obs_flat, actions_flat)
            
            # Critic损失
            critic_loss = F.mse_loss(current_q, y)
            critic_losses[f'agent_{i}'] = critic_loss.item()
            
            # 更新Critic
            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.networks.get_critic_parameters(i),
                self.max_grad_norm
            )
            self.critic_optimizers[i].step()
            
            # ===== 更新Actor =====
            # 获取当前智能体的动作
            current_agent_obs = obs[:, i, :]
            
            # 对于异构观测，需要裁剪到该智能体的实际观测维度
            if hasattr(self.networks, 'heterogeneous_obs') and self.networks.heterogeneous_obs:
                actual_obs_dim = self.networks.obs_dims[i]
                current_agent_obs = current_agent_obs[:, :actual_obs_dim]
            
            current_action_probs = self.networks.actor_networks[i](current_agent_obs)
            
            # 构建用于Critic评估的动作（替换当前智能体的动作）
            actions_for_critic = []
            for j in range(self.n_agents):
                if j == i:
                    actions_for_critic.append(current_action_probs)
                else:
                    agent_obs = obs[:, j, :]
                    
                    # 对于异构观测，需要裁剪到该智能体的实际观测维度
                    if hasattr(self.networks, 'heterogeneous_obs') and self.networks.heterogeneous_obs:
                        actual_obs_dim = self.networks.obs_dims[j]
                        agent_obs = agent_obs[:, :actual_obs_dim]
                    
                    with torch.no_grad():
                        other_action_probs = self.networks.actor_networks[j](agent_obs)
                    actions_for_critic.append(other_action_probs)
            
            actions_for_critic_concat = torch.cat(actions_for_critic, dim=1)
            
            # Actor损失：最大化Q值（因此取负）
            actor_loss = -self.networks.critic_networks[i](obs_flat, actions_for_critic_concat).mean()
            actor_losses[f'agent_{i}'] = actor_loss.item()
            
            # 更新Actor
            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.networks.get_actor_parameters(i),
                self.max_grad_norm
            )
            self.actor_optimizers[i].step()
        
        return critic_losses, actor_losses
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """更新网络"""
        # 计算损失并更新网络
        critic_losses, actor_losses = self.compute_loss(batch)
        
        # 软更新目标网络
        self.networks.soft_update_target_networks(self.tau)
        
        self.update_count += 1
        
        # 返回平均损失
        avg_critic_loss = np.mean(list(critic_losses.values()))
        avg_actor_loss = np.mean(list(actor_losses.values()))
        
        return {
            'critic_loss': avg_critic_loss,
            'actor_loss': avg_actor_loss,
            'total_loss': avg_critic_loss + avg_actor_loss,
            'update_count': self.update_count
        }
    
    def save(self, path: str):
        """保存模型"""
        state_dict = {
            'actor_networks': [net.state_dict() for net in self.networks.actor_networks],
            'critic_networks': [net.state_dict() for net in self.networks.critic_networks],
            'target_actor_networks': [net.state_dict() for net in self.networks.target_actor_networks],
            'target_critic_networks': [net.state_dict() for net in self.networks.target_critic_networks],
            'actor_optimizers': [opt.state_dict() for opt in self.actor_optimizers],
            'critic_optimizers': [opt.state_dict() for opt in self.critic_optimizers],
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
        
        for i, target_actor_net in enumerate(self.networks.target_actor_networks):
            target_actor_net.load_state_dict(state_dict['target_actor_networks'][i])
        
        for i, target_critic_net in enumerate(self.networks.target_critic_networks):
            target_critic_net.load_state_dict(state_dict['target_critic_networks'][i])
        
        for i, actor_opt in enumerate(self.actor_optimizers):
            actor_opt.load_state_dict(state_dict['actor_optimizers'][i])
        
        for i, critic_opt in enumerate(self.critic_optimizers):
            critic_opt.load_state_dict(state_dict['critic_optimizers'][i])
        
        self.update_count = state_dict['update_count']
