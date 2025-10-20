"""
COMA 训练器
基于QMIX trainer结构，实现COMA算法的训练流程
"""
import os
import time
import numpy as np
import torch
import logging
from typing import Dict, Any, List, Optional
import json
from datetime import datetime
from collections import defaultdict

from src.algos import COMA
from src.models import COMANetworks
from src.buffer import ReplayBuffer
from src.utils import set_seed, get_device, load_config
from src.envs import create_ctde_env

logger = logging.getLogger(__name__)


class COMATrainer:
    """COMA训练器"""

    def __init__(self, config):
        """
        初始化训练器

        Args:
            config: 配置字典或配置文件路径
        """
        if isinstance(config, str):
            # 如果是配置文件路径，加载配置
            self.config = load_config(config)
            self.config_path = config
        else:
            # 如果是配置字典，直接使用
            self.config = config
            self.config_path = None

        self.device = get_device()

        # 设置随机种子
        seed = int(time.time())
        set_seed(seed)

        # 创建环境
        self.env_name = self.config['env']['name']
        self.difficulty = self.config['env']['difficulty']
        self.global_state_type = self.config['env'].get('global_state_type', 'concat')

        self.env = create_ctde_env(
            env_name=self.env_name,
            difficulty=self.difficulty,
            global_state_type=self.global_state_type
        )

        # 获取环境信息
        self.env_info = self.env.get_env_info()
        logger.info(f"环境信息: {self.env_info}")

        # 创建网络
        self.networks = COMANetworks(self.env_info, self.config, self.device)

        # 创建算法
        self.coma = COMA(self.networks, self.config, self.device)

        # 创建经验回放缓冲区
        self.buffer = ReplayBuffer(
            capacity=self.config['training']['buffer_size'],
            n_agents=self.env_info['n_agents'],
            obs_dim=max(self.env_info['obs_dims']) if isinstance(self.env_info['obs_dims'], list) else self.env_info['obs_dims'],
            state_dim=self.env_info['global_state_dim'],
            device=self.device
        )

        # 训练参数
        self.total_episodes = self.config['training']['total_episodes']
        self.batch_size = self.config['training']['batch_size']
        self.warmup_episodes = self.config['training']['warmup_episodes']
        self.eval_interval = self.config['training']['eval_interval']
        self.save_interval = self.config['training']['save_interval']

        # 探索参数
        self.epsilon_start = self.config['exploration']['epsilon_start']
        self.epsilon_end = self.config['exploration']['epsilon_end']
        self.epsilon_decay = self.config['exploration']['epsilon_decay']
        self.current_epsilon = self.epsilon_start

        # 训练状态
        self.episode_count = 0
        self.learn_steps = 0

        # 记录指标
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        self.eval_rewards = []
        self.eval_episodes = []
        self.epsilon_history = []

        # 创建保存目录
        self.save_dir = "checkpoints"
        self.plot_dir = "plots"
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)

        logger.info("COMA训练器初始化完成")

    def _update_epsilon(self):
        """更新探索率"""
        self.current_epsilon = max(
            self.epsilon_end,
            self.current_epsilon * self.epsilon_decay
        )
        self.epsilon_history.append(self.current_epsilon)

    def _collect_experience(self) -> Dict[str, float]:
        """收集一个episode的经验"""
        obs, info = self.env.reset()
        episode_reward = 0
        episode_length = 0

        # 获取初始全局状态
        if hasattr(info, 'get') and 'global_state' in info:
            global_state = info['global_state']
        else:
            # 如果没有提供全局状态，使用观测的拼接
            global_state = np.concatenate([obs[agent_id] for agent_id in sorted(obs.keys())])

        done = False

        while not done:
            # 选择动作
            actions = self.coma.select_actions(obs, self.current_epsilon)

            # 执行动作
            step_result = self.env.step(actions)
            if len(step_result) == 4:
                next_obs, rewards, dones, next_info = step_result
                terminated = {agent_id: dones[agent_id] for agent_id in dones}
                truncated = {agent_id: False for agent_id in dones}
            else:
                next_obs, rewards, terminated, truncated, next_info = step_result

            # 获取下一个全局状态
            if hasattr(next_info, 'get') and 'global_state' in next_info:
                next_global_state = next_info['global_state']
            else:
                next_global_state = np.concatenate([next_obs[agent_id] for agent_id in sorted(next_obs.keys())])

            # 判断是否结束
            done = any(terminated.values()) or any(truncated.values())
            dones = {agent_id: terminated[agent_id] or truncated[agent_id] for agent_id in terminated}

            # 存储经验
            self.buffer.push(
                obs=obs,
                actions=actions,
                rewards=rewards,
                next_obs=next_obs,
                dones=dones,
                global_state=global_state,
                next_global_state=next_global_state
            )

            # 更新状态
            obs = next_obs
            global_state = next_global_state

            # 统计
            episode_reward += sum(rewards.values())
            episode_length += 1

        return {
            'episode_reward': episode_reward,
            'episode_length': episode_length
        }

    def _evaluate(self, n_episodes: int = 5) -> float:
        """评估当前策略"""
        total_rewards = []

        for _ in range(n_episodes):
            obs, info = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                # 贪婪策略选择动作
                actions = self.coma.select_actions(obs, epsilon=0.0)

                # 执行动作
                step_result = self.env.step(actions)
                if len(step_result) == 4:
                    next_obs, rewards, dones, next_info = step_result
                    terminated = {agent_id: dones[agent_id] for agent_id in dones}
                    truncated = {agent_id: False for agent_id in dones}
                else:
                    next_obs, rewards, terminated, truncated, next_info = step_result

                # 判断是否结束
                done = any(terminated.values()) or any(truncated.values())

                # 更新状态
                obs = next_obs

                # 统计
                episode_reward += sum(rewards.values())

            total_rewards.append(episode_reward)

        return np.mean(total_rewards)

    def train(self):
        """主训练循环"""
        logger.info(f"开始训练COMA算法")
        logger.info(f"环境: {self.env_name}, 难度: {self.difficulty}")
        logger.info(f"总episodes: {self.total_episodes}")

        start_time = time.time()

        for episode in range(self.total_episodes):
            self.episode_count = episode

            # 收集经验
            episode_stats = self._collect_experience()

            # 记录指标
            self.episode_rewards.append(episode_stats['episode_reward'])
            self.episode_lengths.append(episode_stats['episode_length'])

            # 更新探索率
            self._update_epsilon()

            # 训练网络
            if len(self.buffer) >= self.batch_size and episode >= self.warmup_episodes:
                for _ in range(min(len(self.buffer) // self.batch_size, 10)):  # 每个episode训练多次
                    batch = self.buffer.sample(self.batch_size)
                    train_stats = self.coma.update(batch)
                    self.losses.append(train_stats['actor_loss'] + train_stats['critic_loss'])
                    self.learn_steps += 1

            # 评估
            if episode % self.eval_interval == 0 and episode > 0:
                eval_reward = self._evaluate()
                self.eval_rewards.append(eval_reward)
                self.eval_episodes.append(episode)

                logger.info(f"Episode {episode}: "
                           f"Reward={episode_stats['episode_reward']:.2f}, "
                           f"Eval={eval_reward:.2f}, "
                           f"Epsilon={self.current_epsilon:.3f}, "
                           f"Buffer={len(self.buffer)}")

            # 保存模型
            if episode % self.save_interval == 0 and episode > 0:
                self.save_model(episode)

            # 定期输出进度
            if episode % 100 == 0:
                recent_rewards = self.episode_rewards[-100:] if len(self.episode_rewards) >= 100 else self.episode_rewards
                avg_reward = np.mean(recent_rewards)
                logger.info(f"Episode {episode}: "
                           f"Avg Reward (last 100)={avg_reward:.2f}, "
                           f"Epsilon={self.current_epsilon:.3f}, "
                           f"Learn Steps={self.learn_steps}")

        # 训练完成，保存最终模型和数据
        self.save_model(self.total_episodes)
        self.save_training_data()

        end_time = time.time()
        training_time = end_time - start_time

        logger.info(f"训练完成!")
        logger.info(f"总训练时间: {training_time/3600:.2f} 小时")
        logger.info(f"最终奖励: {self.episode_rewards[-1]:.2f}")
        logger.info(f"平均奖励 (last 100): {np.mean(self.episode_rewards[-100:]):.2f}")

        # 生成图表
        self.generate_plots()

        return self.episode_rewards, self.eval_rewards

    def save_model(self, episode: int):
        """保存模型"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(
            self.save_dir,
            f"COMA_{self.env_name}_{self.difficulty}_episode_{episode}_{timestamp}.pt"
        )
        self.coma.save(model_path)
        logger.info(f"模型已保存到: {model_path}")

    def save_training_data(self):
        """保存训练数据 - 与QMIX格式完全一致"""
        import os
        import json
        import numpy as np
        from datetime import datetime

        os.makedirs(self.save_dir, exist_ok=True)

        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        env_name = self.env_name
        difficulty = self.difficulty

        # 准备数据 - 与QMIX完全相同的结构
        training_data = {
            'config': self.config,
            'metrics': {
                'episode_rewards': self.episode_rewards,
                'episode_lengths': self.episode_lengths,
                'losses': self.losses,
                'eval_episodes': self.eval_episodes,
                'eval_rewards': self.eval_rewards,
                'epsilon_history': self.epsilon_history
            },
            'environment': {
                'name': env_name,
                'difficulty': difficulty,
                'n_agents': self.env_info['n_agents'],
                'obs_dims': self.env_info['obs_dims'],
                'act_dims': self.env_info['act_dims']
            },
            'timestamp': timestamp,
            'total_episodes': len(self.episode_rewards)
        }

        # 保存为JSON文件 - 与QMIX相同的命名格式
        filename = f"{env_name}_{difficulty}_training_data_{timestamp}.json"
        filepath = os.path.join(self.save_dir, filename)

        # 转换numpy类型为Python原生类型 - 与QMIX相同的转换逻辑
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj

        training_data_converted = convert_numpy(training_data)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(training_data_converted, f, indent=2, ensure_ascii=False)

        print(f"💾 训练数据已保存: {filepath}")
        return filepath

    def generate_plots(self):
        """生成训练图表"""
        try:
            from .utils import save_training_plots

            env_name = f"COMA_{self.env_name}_{self.difficulty}"
            save_training_plots(
                episode_rewards=self.episode_rewards,
                episode_lengths=self.episode_lengths,
                losses=self.losses,
                eval_episodes=self.eval_episodes,
                eval_rewards=self.eval_rewards,
                epsilon_values=self.epsilon_history,
                save_dir=self.plot_dir,
                env_name=env_name,
                config=self.config,
                show_plots=False
            )

            logger.info(f"训练图表已保存到: {self.plot_dir}")

        except ImportError as e:
            logger.warning(f"无法生成图表: {e}")
        except Exception as e:
            logger.error(f"生成图表时发生错误: {e}")