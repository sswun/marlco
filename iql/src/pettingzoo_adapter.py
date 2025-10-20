"""
PettingZoo环境适配器 - 将PettingZoo环境适配为统一的CTDE接口
"""
import logging
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
import gymnasium as gym
from pettingzoo.sisl import multiwalker_v9
from pettingzoo.mpe import simple_spread_v3, simple_crypto_v3


class PettingZooAdapter:
    """PettingZoo环境适配器"""

    def __init__(self, env_name: str, difficulty: str = "default", global_state_type: str = "concat", **kwargs):
        self.env_name = env_name
        self.difficulty = difficulty
        self.global_state_type = global_state_type
        self.kwargs = kwargs

        # 创建PettingZoo环境
        self.pettingzoo_env = self._create_pettingzoo_env()

        # 获取环境信息
        self.possible_agents = self.pettingzoo_env.possible_agents
        self.n_agents = len(self.possible_agents)

        # 离散化连续动作空间
        self.action_discretizer = self._create_action_discretizer()

    def _create_pettingzoo_env(self):
        """创建PettingZoo环境"""
        if self.env_name == "multiwalker":
            # Multiwalker环境 - 连续动作空间
            env = multiwalker_v9.parallel_env(
                n_walkers=3,
                position_noise=1e-3,
                angle_noise=1e-3,
                forward_reward=1.0,
                terminate_reward=-100.0,
                fall_reward=-10.0,
                shared_reward=True,
                terminate_on_fall=True,
                remove_on_fall=True,
                terrain_length=200,
                max_cycles=500,
                render_mode=None
            )
        elif self.env_name == "simple_spread":
            # Simple Spread环境
            env = simple_spread_v3.parallel_env(
                N=3,
                local_ratio=0.5,
                max_cycles=25,
                continuous_actions=False  # 使用离散动作
            )
        elif self.env_name == "simple_crypto":
            # Simple Crypto环境
            env = simple_crypto_v3.parallel_env(
                max_cycles=25,
                continuous_actions=False  # 使用离散动作
            )
        else:
            raise ValueError(f"Unsupported PettingZoo environment: {self.env_name}")

        return env

    def _create_action_discretizer(self):
        """为连续动作空间创建离散化器"""
        if self.env_name == "multiwalker":
            # Multiwalker有4维连续动作空间，每维范围[-1, 1]
            # 将每维离散化为5个值：-1, -0.5, 0, 0.5, 1
            n_actions_per_dim = 5
            action_values = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
            total_actions = n_actions_per_dim ** 4  # 5^4 = 625

            def discrete_to_continuous(discrete_action):
                """将离散动作转换为连续动作"""
                # 将整数转换为4维索引
                action_4d = []
                remaining = discrete_action
                for i in range(4):
                    action_4d.append(remaining % n_actions_per_dim)
                    remaining = remaining // n_actions_per_dim
                # 转换为连续值
                continuous_action = [action_values[idx] for idx in action_4d]
                return continuous_action

            return {
                'total_actions': total_actions,
                'discrete_to_continuous': discrete_to_continuous,
                'is_continuous': True
            }
        else:
            # Simple Spread和Simple Crypto已经是离散的
            return {
                'total_actions': None,  # 将在reset时设置
                'discrete_to_continuous': lambda x: x,
                'is_continuous': False
            }

    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """重置环境"""
        observations, infos = self.pettingzoo_env.reset()

        # 保存当前观测用于全局状态计算
        self._current_obs = observations

        # 更新离散化器的动作数（对于离散环境）
        if not self.action_discretizer['is_continuous']:
            # 获取第一个智能体的动作空间大小
            first_agent = self.possible_agents[0]
            action_space = self.pettingzoo_env.action_space(first_agent)
            self.action_discretizer['total_actions'] = action_space.n

        return observations, infos

    def step(self, actions: Dict[str, int]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, Any]]:
        """执行一步"""
        # 转换动作为PettingZoo格式
        pettingzoo_actions = {}
        for agent_id, action in actions.items():
            if self.action_discretizer['is_continuous']:
                # 转换离散动作为连续动作
                continuous_action = self.action_discretizer['discrete_to_continuous'](action)
                pettingzoo_actions[agent_id] = continuous_action
            else:
                pettingzoo_actions[agent_id] = action

        # 执行步骤
        observations, rewards, terminated, truncated, infos = self.pettingzoo_env.step(pettingzoo_actions)

        # 合并terminated和truncated
        dones = {}
        for agent_id in observations.keys():
            dones[agent_id] = terminated[agent_id] or truncated[agent_id]

        # 保存当前观测用于全局状态计算
        self._current_obs = observations

        # 添加全局状态到infos
        infos['global_state'] = self.get_global_state()

        return observations, rewards, dones, infos

    def get_global_state(self) -> np.ndarray:
        """获取全局状态"""
        try:
            if hasattr(self.pettingzoo_env, 'state'):
                state = self.pettingzoo_env.state()
                if state is not None:
                    # MultiWalker环境：固定状态维度以处理动态智能体数量
                    if self.env_name == "multiwalker":
                        # 使用最大智能体数量来固定维度
                        max_agents = 3  # multiwalker默认3个智能体
                        state_per_agent = 25  # 每个智能体的状态维度
                        expected_dim = max_agents * state_per_agent  # 75

                        if state.shape[0] < expected_dim:
                            # 智能体减少了，填充0
                            padded_state = np.zeros(expected_dim, dtype=state.dtype)
                            padded_state[:state.shape[0]] = state
                            return padded_state
                        else:
                            # 截断到固定维度
                            return state[:expected_dim]
                    return state
        except Exception as e:
            # state()方法失败，回退到拼接观测
            pass

        # 如果没有state方法或调用失败，通过拼接当前观测来构造全局状态
        # 这需要一个当前观测，如果没有则返回空数组
        if hasattr(self, '_current_obs') and self._current_obs is not None:
            observations = self._current_obs
            if isinstance(observations, dict):
                # 拼接所有智能体的观测
                all_obs = []
                for agent_id in self.possible_agents:
                    if agent_id in observations:
                        obs = observations[agent_id]
                        if isinstance(obs, np.ndarray):
                            all_obs.append(obs.flatten())
                        else:
                            all_obs.append([obs])

                if all_obs:
                    concatenated = np.concatenate(all_obs)

                    # MultiWalker环境：固定维度
                    if self.env_name == "multiwalker":
                        max_agents = 3
                        obs_per_agent = 31  # 每个智能体观测维度
                        expected_dim = max_agents * obs_per_agent  # 93

                        if concatenated.shape[0] < expected_dim:
                            padded_obs = np.zeros(expected_dim, dtype=concatenated.dtype)
                            padded_obs[:concatenated.shape[0]] = concatenated
                            return padded_obs
                        else:
                            return concatenated[:expected_dim]

                    return concatenated
                else:
                    return np.array([])

        # 如果所有方法都失败，返回空数组
        return np.array([])

    def get_env_info(self) -> Dict[str, Any]:
        """获取环境信息"""
        # 需要重置环境来获取准确的空间信息
        temp_obs, _ = self.pettingzoo_env.reset()

        # 按照算法中使用的排序顺序排列智能体
        sorted_agent_ids = sorted(self.possible_agents)

        # 获取每个智能体的观测和动作空间信息
        obs_dims = []
        act_dims = []

        for agent_id in sorted_agent_ids:
            obs_space = self.pettingzoo_env.observation_space(agent_id)
            act_space = self.pettingzoo_env.action_space(agent_id)

            # 观测维度
            if isinstance(obs_space, gym.spaces.Box):
                obs_dim = obs_space.shape[0] if len(obs_space.shape) == 1 else np.prod(obs_space.shape)
            else:
                obs_dim = 1  # 离散观测空间
            obs_dims.append(obs_dim)

            # 动作维度
            if hasattr(act_space, 'n'):
                # 离散动作空间
                act_dim = act_space.n
            elif isinstance(act_space, gym.spaces.Box) and self.action_discretizer['is_continuous']:
                # 连续动作空间,使用离散化器
                act_dim = self.action_discretizer['total_actions']
            else:
                raise ValueError(f"Unsupported action space type: {type(act_space)}")
            act_dims.append(act_dim)

        # 设置离散化器的动作数
        if self.action_discretizer['total_actions'] is None:
            self.action_discretizer['total_actions'] = act_dims[0]  # 假设所有智能体动作空间相同

        # 全局状态维度 - 实际的智能体观测维度之和
        global_state_dim = sum(obs_dims)

        info = {
            'n_agents': self.n_agents,
            'agent_ids': sorted_agent_ids,  # 使用排序后的智能体ID
            'obs_dims': obs_dims,  # 使用排序后的观测维度
            'act_dims': act_dims,  # 使用排序后的动作维度
            'global_state_dim': global_state_dim,
            'episode_limit': 500 if self.env_name == "multiwalker" else 25,
            'env_name': self.env_name
        }

        return info

    def get_avail_actions(self, agent_id: str) -> np.ndarray:
        """获取智能体可用动作 - 返回可用动作的索引数组"""
        total_actions = self.action_discretizer['total_actions']
        if total_actions is None:
            # 在reset之后设置
            return np.array([])

        # 所有动作都可用，返回所有动作的索引
        return np.arange(total_actions)

    def close(self):
        """关闭环境"""
        if hasattr(self.pettingzoo_env, 'close'):
            self.pettingzoo_env.close()


def create_pettingzoo_adapter(env_name: str, difficulty: str = "default",
                            global_state_type: str = "concat", **kwargs) -> PettingZooAdapter:
    """创建PettingZoo环境适配器"""
    return PettingZooAdapter(env_name, difficulty, global_state_type, **kwargs)