"""
MAPPO 算法核心实现
"""
import torch
import torch.nn.functional as F
import torch.optim
import numpy as np
from typing import Dict, Any, Tuple, List
from .models import MAPPONetworks


class RolloutBuffer:
    """PPO回放缓冲区"""
    
    def __init__(self, buffer_size: int, n_agents: int, obs_dim: int, 
                 state_dim: int, device='cpu'):
        self.buffer_size = buffer_size
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.device = device
        
        # 存储数据
        self.obs = []
        self.states = []
        self.actions = []
        self.action_log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
        self.ptr = 0
        self.episode_start = 0
        
    def push(self, obs, state, action, action_log_prob, reward, value, done):
        """添加一步数据"""
        self.obs.append(obs)
        self.states.append(state)
        self.actions.append(action)
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        
        self.ptr += 1
        
    def get(self):
        """获取所有数据"""
        return {
            'obs': self.obs,
            'states': self.states,
            'actions': self.actions,
            'action_log_probs': self.action_log_probs,
            'rewards': self.rewards,
            'values': self.values,
            'dones': self.dones
        }
    
    def clear(self):
        """清空缓冲区"""
        self.obs = []
        self.states = []
        self.actions = []
        self.action_log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.ptr = 0
        
    def __len__(self):
        return len(self.obs)


class MAPPO:
    """MAPPO算法实现"""
    
    def __init__(self, networks: MAPPONetworks, config: Dict[str, Any], device='cpu'):
        self.networks = networks
        self.config = config
        self.device = device
        self.n_agents = networks.n_agents
        
        # 算法参数
        self.gamma = config['algorithm']['gamma']
        self.gae_lambda = config['algorithm']['gae_lambda']
        self.clip_param = config['algorithm']['clip_param']
        self.value_loss_coef = config['algorithm']['value_loss_coef']
        self.entropy_coef = config['algorithm']['entropy_coef']
        self.max_grad_norm = config['algorithm']['max_grad_norm']
        self.ppo_epochs = config['algorithm']['ppo_epochs']
        self.num_mini_batch = config['algorithm']['num_mini_batch']
        
        # 为每个智能体创建Actor优化器
        self.actor_optimizers = []
        for i in range(self.n_agents):
            actor_optimizer = torch.optim.Adam(
                self.networks.get_actor_parameters(i),
                lr=config['algorithm']['actor_lr']
            )
            self.actor_optimizers.append(actor_optimizer)
        
        # Critic优化器（共享）
        self.critic_optimizer = torch.optim.Adam(
            self.networks.get_critic_parameters(),
            lr=config['algorithm']['critic_lr']
        )
        
        self.update_count = 0
    
    def select_actions(self, obs: Dict[str, torch.Tensor], 
                      deterministic: bool = False,
                      avail_actions: Dict[str, list] = None) -> Tuple[Dict[str, int], Dict[str, float]]:
        """
        选择动作
        
        Args:
            obs: 观测字典
            deterministic: 是否确定性选择
            avail_actions: 可用动作字典
            
        Returns:
            actions: 动作字典
            action_log_probs: 动作对数概率字典
        """
        actions = {}
        action_log_probs = {}
        
        # 获取排序的agent_ids
        agent_ids = sorted(obs.keys())
        
        for i, agent_id in enumerate(agent_ids):
            agent_obs = obs[agent_id].unsqueeze(0) if obs[agent_id].dim() == 1 else obs[agent_id]

            # 对于异构观测，需要裁剪到该智能体的实际观测维度
            if hasattr(self.networks, 'heterogeneous_obs') and self.networks.heterogeneous_obs:
                actual_obs_dim = self.networks.obs_dims[i]
                agent_obs = agent_obs[:, :actual_obs_dim]

            with torch.no_grad():
                # 获取动作logits
                logits = self.networks.actor_networks[i](agent_obs)
                
                # 数值稳定性：使用log_softmax技巧，避免直接计算softmax
                # 不硬裁剪logits，而是使用数值稳定的计算方法
                log_probs = F.log_softmax(logits, dim=-1)
                probs = torch.exp(log_probs)
                
                # 检查数值有效性（只在极端情况下干预）
                if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs < 0).any():
                    # 极端情况：使用均匀分布作为降级策略（静默处理，不打印）
                    probs = torch.ones_like(probs) / probs.shape[-1]
                    log_probs = torch.log(probs)
                
                if deterministic:
                    # 确定性选择（取最大概率）
                    action = probs.argmax(dim=-1)
                else:
                    # 随机采样
                    action = torch.multinomial(probs, num_samples=1).squeeze(-1)
                
                # 计算对数概率
                log_probs = F.log_softmax(logits, dim=-1)
                # 处理维度问题：确保action和log_probs维度匹配
                if action.dim() == 0:  # 标量
                    # log_probs shape: (1, action_dim), action是标量
                    action_log_prob = log_probs[0, action]
                else:  # 批量
                    action_log_prob = log_probs.gather(1, action.unsqueeze(-1)).squeeze(-1)
                
                # 如果提供了可用动作，确保选择有效动作
                if avail_actions is not None and agent_id in avail_actions and avail_actions[agent_id] is not None:
                    avail = avail_actions[agent_id]
                    if action.item() not in avail:
                        action = torch.tensor(avail[torch.randint(0, len(avail), (1,)).item()], 
                                            device=action.device)
                        # 重新计算对数概率
                        if action.dim() == 0:  # 标量
                            # log_probs shape: (1, action_dim), action是标量
                            action_log_prob = log_probs[0, action]
                        else:  # 批量
                            action_log_prob = log_probs.gather(1, action.unsqueeze(-1)).squeeze(-1)
                
                actions[agent_id] = action.item()
                action_log_probs[agent_id] = action_log_prob.item()
        
        return actions, action_log_probs
    
    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """获取状态价值"""
        with torch.no_grad():
            value = self.networks.critic_network(state)
        return value
    
    def compute_gae(self, rewards: List, values: List, dones: List, 
                    next_value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算广义优势估计(GAE)
        
        Args:
            rewards: 奖励列表
            values: 价值列表
            dones: 终止标志列表
            next_value: 下一状态价值
            
        Returns:
            returns: 回报
            advantages: 优势
        """
        advantages = []
        gae = 0
        
        # 转换为tensor
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # 从后向前计算GAE
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        advantages = torch.stack(advantages).squeeze()
        returns = advantages + torch.stack(values).squeeze()
        
        return returns, advantages
    
    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """PPO更新"""
        # 获取缓冲区数据
        data = buffer.get()
        
        if len(data['obs']) == 0:
            return {'actor_loss': 0.0, 'critic_loss': 0.0, 'entropy': 0.0}
        
        # 准备数据
        batch_size = len(data['obs'])
        
        # 转换为tensor并合并
        obs_batch = []
        states_batch = []
        actions_batch = []
        old_action_log_probs_batch = []
        rewards_batch = []
        values_batch = []
        dones_batch = []
        
        for i in range(batch_size):
            # 每个时间步的数据
            obs_dict = data['obs'][i]
            state = data['states'][i]
            action_dict = data['actions'][i]
            log_prob_dict = data['action_log_probs'][i]
            reward = sum(data['rewards'][i].values())  # 团队奖励
            value = data['values'][i]
            done = all(data['dones'][i].values())
            
            # 将字典转换为数组
            agent_ids = sorted(obs_dict.keys())
            
            # 处理异构观测：使用padding确保维度一致
            if hasattr(self.networks, 'heterogeneous_obs') and self.networks.heterogeneous_obs:
                # 每个智能体的观测需要padding到最大维度
                max_obs_dim = self.networks.obs_dim
                obs_list = []
                for aid in agent_ids:
                    obs = obs_dict[aid]
                    if len(obs) < max_obs_dim:
                        # Padding到最大维度
                        padded_obs = np.zeros(max_obs_dim)
                        padded_obs[:len(obs)] = obs
                        obs_list.append(padded_obs)
                    else:
                        obs_list.append(obs)
                obs_array = np.array(obs_list)
            else:
                # 同构观测，直接转换
                obs_array = np.array([obs_dict[aid] for aid in agent_ids])
            
            action_array = np.array([action_dict[aid] for aid in agent_ids])
            log_prob_array = np.array([log_prob_dict[aid] for aid in agent_ids])
            
            obs_batch.append(obs_array)
            states_batch.append(state)
            actions_batch.append(action_array)
            old_action_log_probs_batch.append(log_prob_array)
            rewards_batch.append(reward)
            values_batch.append(value)
            dones_batch.append(float(done))
        
        # 转换为tensor
        states_batch = torch.FloatTensor(np.array(states_batch)).to(self.device)
        values_batch = [torch.FloatTensor([v]).to(self.device) for v in values_batch]
        
        # 计算下一状态价值（用于GAE）
        if dones_batch[-1]:
            next_value = torch.zeros(1).to(self.device)
        else:
            next_value = self.get_value(states_batch[-1:])
        
        # 计算GAE
        returns, advantages = self.compute_gae(rewards_batch, values_batch, dones_batch, next_value)
        
        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO更新
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        
        for epoch in range(self.ppo_epochs):
            # 对于每个智能体
            for agent_idx in range(self.n_agents):
                # 准备该智能体的数据
                agent_obs = torch.FloatTensor(np.array([obs_batch[t][agent_idx] for t in range(batch_size)])).to(self.device)
                agent_actions = torch.LongTensor(np.array([actions_batch[t][agent_idx] for t in range(batch_size)])).to(self.device)
                agent_old_log_probs = torch.FloatTensor(np.array([old_action_log_probs_batch[t][agent_idx] for t in range(batch_size)])).to(self.device)
                
                # 对于异构观测，需要裁剪
                if hasattr(self.networks, 'heterogeneous_obs') and self.networks.heterogeneous_obs:
                    actual_obs_dim = self.networks.obs_dims[agent_idx]
                    agent_obs = agent_obs[:, :actual_obs_dim]
                
                # 评估动作
                new_action_log_probs, dist_entropy = self.networks.actor_networks[agent_idx].evaluate_actions(
                    agent_obs, agent_actions
                )
                
                # 计算比率
                ratio = torch.exp(new_action_log_probs - agent_old_log_probs)
                
                # PPO裁剪
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
                
                # Actor损失
                actor_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -self.entropy_coef * dist_entropy
                
                # 更新Actor
                self.actor_optimizers[agent_idx].zero_grad()
                (actor_loss + entropy_loss).backward()
                torch.nn.utils.clip_grad_norm_(
                    self.networks.get_actor_parameters(agent_idx),
                    self.max_grad_norm
                )
                self.actor_optimizers[agent_idx].step()
                
                total_actor_loss += actor_loss.item()
                total_entropy += dist_entropy.item()
            
            # 更新Critic（共享）
            values_pred = self.networks.critic_network(states_batch).squeeze()
            critic_loss = F.mse_loss(values_pred, returns)
            
            self.critic_optimizer.zero_grad()
            (self.value_loss_coef * critic_loss).backward()
            torch.nn.utils.clip_grad_norm_(
                self.networks.get_critic_parameters(),
                self.max_grad_norm
            )
            self.critic_optimizer.step()
            
            total_critic_loss += critic_loss.item()
        
        self.update_count += 1
        
        # 返回平均损失
        num_updates = self.ppo_epochs * self.n_agents
        return {
            'actor_loss': total_actor_loss / num_updates,
            'critic_loss': total_critic_loss / self.ppo_epochs,
            'entropy': total_entropy / num_updates,
            'update_count': self.update_count
        }
    
    def save(self, path: str):
        """保存模型"""
        state_dict = {
            'actor_networks': [net.state_dict() for net in self.networks.actor_networks],
            'critic_network': self.networks.critic_network.state_dict(),
            'actor_optimizers': [opt.state_dict() for opt in self.actor_optimizers],
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
        
        self.networks.critic_network.load_state_dict(state_dict['critic_network'])
        
        for i, actor_opt in enumerate(self.actor_optimizers):
            actor_opt.load_state_dict(state_dict['actor_optimizers'][i])
        
        self.critic_optimizer.load_state_dict(state_dict['critic_optimizer'])
        self.update_count = state_dict['update_count']
