"""
COMA 环境包装器 - 统一CTDE环境接口
与QMIX envs保持一致
"""
import sys
import os
import logging
import numpy as np
from typing import Dict, Any, Tuple, Optional

# 添加Env路径到sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../Env'))

# 添加qmix路径到sys.path以使用PettingZoo适配器
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../qmix'))

# PettingZoo适配器将在需要时从QMIX导入


class EnvWrapper:
    """统一的环境包装器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.env_name = config['env']['name']
        self.difficulty = config['env']['difficulty']
        self.global_state_type = config['env'].get('global_state_type', 'concat')

        # 创建环境
        self.env = self._create_env()
        self.env_info = self.env.get_env_info()

        # 统一接口
        self.n_agents = self.env_info['n_agents']
        self.agent_ids = self.env_info.get('agent_ids', [f'agent_{i}' for i in range(self.n_agents)])

    def _create_env(self):
        """根据配置创建环境"""
        if self.env_name == 'DEM':
            # 禁用DEM环境的日志输出
            logging.getLogger('DEM.env_dem').setLevel(logging.WARNING)
            logging.getLogger('DEM.env_dem_ctde').setLevel(logging.WARNING)

            from DEM.env_dem_ctde import create_dem_ctde_env
            return create_dem_ctde_env(
                difficulty=self.difficulty,
                global_state_type=self.global_state_type,
                render_mode=""  # 禁用渲染以加快训练速度
            )
        elif self.env_name == 'HRG':
            # 禁用HRG环境的日志输出
            logging.getLogger('HRG.env_hrg').setLevel(logging.WARNING)
            logging.getLogger('HRG.env_hrg_ctde').setLevel(logging.WARNING)
            logging.getLogger('HRG.env_hrg_fast').setLevel(logging.WARNING)
            logging.getLogger('HRG.env_hrg_fast_ctde').setLevel(logging.WARNING)
            logging.getLogger('HRG.env_hrg_ultra_fast').setLevel(logging.WARNING)
            logging.getLogger('HRG.env_hrg_ultra_fast_ctde').setLevel(logging.WARNING)

            # 检查是否使用快速版本
            use_fast = self.difficulty.startswith('fast') or self.difficulty.startswith('ultra_fast')

            if self.difficulty == 'ultra_fast' or self.difficulty == 'ultra_fast_ctde':
                # 使用超快速版本
                from HRG.env_hrg_ultra_fast_ctde import create_hrg_ultra_fast_ctde_env
                return create_hrg_ultra_fast_ctde_env()
            elif use_fast:
                # 使用快速优化版本
                from HRG.env_hrg_fast_ctde import create_hrg_fast_ctde_env
                return create_hrg_fast_ctde_env(difficulty=self.difficulty)
            else:
                # 使用标准版本
                from HRG.env_hrg_ctde import create_hrg_ctde_env
                return create_hrg_ctde_env(
                    config_name=self.difficulty + '_ctde',
                    global_state_type=self.global_state_type,
                    render_mode=""  # 禁用渲染以加快训练速度
                )
        elif self.env_name == 'MSFS':
            # 禁用MSFS环境的日志输出
            logging.getLogger('MSFS.env_msfs').setLevel(logging.WARNING)
            logging.getLogger('MSFS.env_msfs_ctde').setLevel(logging.WARNING)

            from MSFS.env_msfs_ctde import create_msfs_ctde_env

            return create_msfs_ctde_env(
                difficulty=self.difficulty,
                global_state_type=self.global_state_type,
                render_mode=""  # 禁用渲染以加快训练速度
            )
        elif self.env_name == 'CM':
            # 禁用CM环境的日志输出
            logging.getLogger('CM.env_cm').setLevel(logging.WARNING)
            logging.getLogger('CM.env_cm_ctde').setLevel(logging.WARNING)

            from CM.env_cm_ctde import create_cm_ctde_env
            return create_cm_ctde_env(
                difficulty=self.difficulty + '_ctde',
                global_state_type=self.global_state_type,
                render_mode=""  # 禁用渲染以加快训练速度
            )
        elif self.env_name == 'SMAC':
            # 禁用SMAC环境的日志输出
            logging.getLogger('SMAC.env_smac').setLevel(logging.WARNING)
            logging.getLogger('SMAC.env_smac_ctde').setLevel(logging.WARNING)

            from SMAC.env_smac_ctde import create_smac_ctde_env
            # SMAC使用map_name而不是difficulty
            # 如果difficulty是预定义的配置名，尝试使用；否则作为map_name
            if self.difficulty in ['easy', 'normal', 'hard', 'debug']:
                from SMAC.config import get_config_by_name
                from SMAC.env_smac import SMACEnv
                config = get_config_by_name(self.difficulty)
                config.render_mode = ""  # 禁用渲染
                base_env = SMACEnv(config)
                from SMAC.env_smac_ctde import SMACCTDEEnv
                return SMACCTDEEnv(base_env)
            else:
                # 将difficulty作为map_name使用
                return create_smac_ctde_env(
                    map_name=self.difficulty,
                    render_mode=""
                )
        elif self.env_name in ['multiwalker', 'simple_spread', 'simple_crypto']:
            # PettingZoo环境 - 与QMIX/VDN保持一致
            logging.getLogger('pettingzoo').setLevel(logging.WARNING)
            # 复用QMIX的PettingZoo适配器以保持一致性
            try:
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../qmix/src'))
                from pettingzoo_adapter import create_pettingzoo_adapter
                return create_pettingzoo_adapter(
                    env_name=self.env_name,
                    difficulty=self.difficulty,
                    global_state_type=self.global_state_type
                )
            except ImportError:
                # 如果无法导入QMIX适配器，创建简化版本
                return self._create_pettingzoo_adapter_fallback()
        else:
            raise ValueError(f"Unsupported environment: {self.env_name}")

    def reset(self) -> Tuple[Dict[str, np.ndarray], Optional[Dict[str, Any]]]:
        """重置环境"""
        obs = self.env.reset()

        # 处理不同环境的返回格式
        if isinstance(obs, tuple):
            obs = obs[0]  # 取观测部分

        # 确保返回字典格式
        if not isinstance(obs, dict):
            obs = {agent_id: obs[i] for i, agent_id in enumerate(self.agent_ids)}

        return obs, None

    def step(self, actions: Dict[str, int]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, Any]]:
        """执行一步"""
        # 将actions转换为环境需要的格式
        if hasattr(self.env, 'step'):
            step_result = self.env.step(actions)
        else:
            raise RuntimeError("Environment does not have step method")

        # 处理不同环境的返回格式
        if len(step_result) == 4:
            obs, rewards, dones, infos = step_result
        elif len(step_result) == 5:
            obs, rewards, dones, truncated, infos = step_result
            # 合并dones和truncated
            if isinstance(dones, dict) and isinstance(truncated, dict):
                for agent_id in dones.keys():
                    dones[agent_id] = dones[agent_id] or truncated.get(agent_id, False)
        else:
            raise ValueError(f"Unexpected step result length: {len(step_result)}")

        # 确保返回字典格式
        if not isinstance(obs, dict):
            obs = {agent_id: obs[i] for i, agent_id in enumerate(self.agent_ids)}
        if not isinstance(rewards, dict):
            rewards = {agent_id: rewards[i] for i, agent_id in enumerate(self.agent_ids)}
        if not isinstance(dones, dict):
            dones = {agent_id: dones[i] if hasattr(dones, '__getitem__') else dones
                    for i, agent_id in enumerate(self.agent_ids)}

        # 添加全局状态到info
        if 'global_state' not in infos:
            infos['global_state'] = self.get_global_state()

        return obs, rewards, dones, infos

    def get_global_state(self) -> np.ndarray:
        """获取全局状态"""
        return self.env.get_global_state()

    def get_avail_actions(self, agent_id: str):
        """获取智能体可用动作（如果环境支持）"""
        if hasattr(self.env, 'get_avail_actions'):
            return self.env.get_avail_actions(agent_id)
        return None

    def get_env_info(self) -> Dict[str, Any]:
        """获取环境信息"""
        info = self.env_info.copy()

        # 标准化信息格式 - 统一转换为列表格式
        # 处理 obs_dims
        if 'obs_dims' in info:
            if isinstance(info['obs_dims'], dict):
                # 字典格式 -> 列表格式 (CM, SMAC环境)
                info['obs_dims'] = [info['obs_dims'][agent_id] for agent_id in self.agent_ids]
            elif isinstance(info['obs_dims'], list):
                # 已经是列表格式 (HRG环境)
                pass
            else:
                # 其他格式，转换为列表
                info['obs_dims'] = [info['obs_dims']] * self.n_agents
        elif 'obs_shape' in info:
            # 通过 obs_shape 生成 obs_dims (DEM, MSFS环境)
            if isinstance(info['obs_shape'], int):
                info['obs_dims'] = [info['obs_shape']] * self.n_agents
            elif isinstance(info['obs_shape'], tuple):
                info['obs_dims'] = [info['obs_shape'][0]] * self.n_agents
            else:
                info['obs_dims'] = [info['obs_shape']] * self.n_agents

        # 处理 act_dims
        if 'act_dims' in info:
            if isinstance(info['act_dims'], dict):
                # 字典格式 -> 列表格式 (CM, SMAC环境)
                info['act_dims'] = [info['act_dims'][agent_id] for agent_id in self.agent_ids]
            elif isinstance(info['act_dims'], list):
                # 已经是列表格式 (HRG环境)
                pass
            else:
                # 其他格式，转换为列表
                info['act_dims'] = [info['act_dims']] * self.n_agents
        elif 'n_actions' in info:
            # 通过 n_actions 生成 act_dims (DEM, MSFS环境)
            info['act_dims'] = [info['n_actions']] * self.n_agents

        # 确保 agent_ids 存在
        if 'agent_ids' not in info:
            info['agent_ids'] = self.agent_ids

        return info

    def _create_pettingzoo_adapter_fallback(self):
        """创建PettingZoo适配器的fallback版本"""
        try:
            import pettingzoo
            from pettingzoo.mpe import simple_spread_v3, simple_crypto_v3
            from pettingzoo.sisl import multiwalker_v9
            import gymnasium as gym
            import numpy as np

            if self.env_name == "simple_spread":
                env = simple_spread_v3.parallel_env(max_cycles=25, continuous_actions=False)
            elif self.env_name == "simple_crypto":
                env = simple_crypto_v3.parallel_env(max_cycles=25, continuous_actions=False)
            elif self.env_name == "multiwalker":
                # 创建与QMIX一致的多walker环境
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
            else:
                raise ValueError(f"Unsupported PettingZoo environment: {self.env_name}")

            # 创建适配器
            class PettingZooFallbackAdapter:
                def __init__(self, env, env_name):
                    self.env = env
                    self.env_name = env_name
                    self.possible_agents = env.possible_agents
                    self.n_agents = len(self.possible_agents)

                    # 设置动作离散化器（仅用于multiwalker）
                    if self.env_name == "multiwalker":
                        n_actions_per_dim = 5
                        action_values = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
                        self.total_actions = n_actions_per_dim ** 4  # 5^4 = 625
                        self.action_values = action_values
                        self.n_actions_per_dim = n_actions_per_dim

                def _discrete_to_continuous(self, discrete_action):
                    """将离散动作转换为连续动作（仅用于multiwalker）"""
                    if self.env_name != "multiwalker":
                        return discrete_action

                    # 将整数转换为4维索引
                    action_4d = []
                    remaining = discrete_action
                    for i in range(4):
                        action_4d.append(remaining % self.n_actions_per_dim)
                        remaining = remaining // self.n_actions_per_dim
                    # 转换为连续值
                    continuous_action = [self.action_values[idx] for idx in action_4d]
                    return continuous_action

                def reset(self):
                    obs, infos = self.env.reset()
                    # 转换为字典格式
                    return obs, {'global_state': self._get_global_state(obs)}

                def step(self, actions):
                    # 转换动作（如果需要）
                    if self.env_name == "multiwalker":
                        pettingzoo_actions = {}
                        for agent_id, action in actions.items():
                            continuous_action = self._discrete_to_continuous(action)
                            pettingzoo_actions[agent_id] = continuous_action
                    else:
                        pettingzoo_actions = actions

                    obs, rewards, terminateds, truncateds, infos = self.env.step(pettingzoo_actions)
                    done = all(terminateds.values()) or all(truncateds.values())

                    # 转换为字典格式
                    obs_dict = obs
                    rewards_dict = rewards
                    dones_dict = {agent: terminateds[agent] or truncateds[agent] for agent in terminateds}
                    infos_dict = {'global_state': self._get_global_state(obs)}

                    return obs_dict, rewards_dict, dones_dict, infos_dict

                def get_global_state(self):
                    # 简单的全局状态：拼接所有观测
                    obs, _ = self.env.reset()
                    return self._get_global_state(obs)

                def _get_global_state(self, obs):
                    # 拼接所有智能体的观测
                    state = np.concatenate([obs[agent] for agent in self.possible_agents])
                    return state

                def get_env_info(self):
                    sample_obs, _ = self.env.reset()

                    # 检查每个智能体的观测维度
                    obs_dims = []
                    for agent in self.possible_agents:
                        obs_dim = sample_obs[agent].shape[0]
                        obs_dims.append(obs_dim)

                    # 动作维度
                    if self.env_name == "multiwalker":
                        action_dim = 625  # 5^4
                    else:
                        sample_agent = self.possible_agents[0]
                        action_space = self.env.action_space(sample_agent)
                        action_dim = action_space.n

                    return {
                        'n_agents': self.n_agents,
                        'agent_ids': self.possible_agents,
                        'obs_dims': obs_dims,
                        'act_dims': [action_dim] * self.n_agents,
                        'global_state_dim': sum(obs_dims),
                        'episode_limit': 500 if self.env_name == "multiwalker" else 25,
                        'discrete_actions': True
                    }

                def close(self):
                    self.env.close()

            return PettingZooFallbackAdapter(env, self.env_name)

        except ImportError:
            raise ImportError("PettingZoo not available. Please install pettingzoo package: pip install pettingzoo[mpe]")

    def close(self):
        """关闭环境"""
        if hasattr(self.env, 'close'):
            self.env.close()


def create_env_wrapper(config: Dict[str, Any]) -> EnvWrapper:
    """创建环境包装器的工厂函数"""
    return EnvWrapper(config)


def create_ctde_env(env_name: str, difficulty: str, global_state_type: str = 'concat') -> EnvWrapper:
    """
    创建CTDE环境的便捷函数

    Args:
        env_name: 环境名称
        difficulty: 难度设置
        global_state_type: 全局状态类型

    Returns:
        环境包装器实例
    """
    config = {
        'env': {
            'name': env_name,
            'difficulty': difficulty,
            'global_state_type': global_state_type
        }
    }
    return create_env_wrapper(config)