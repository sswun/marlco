"""
VDN 环境包装器 - 统一CTDE环境接口
"""
import sys
import os
import logging
import numpy as np
from typing import Dict, Any, Tuple, Optional, List

# 添加Env路径到sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../Env'))

# 导入PettingZoo适配器（从QMIX复用）
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../qmix/src'))
    from pettingzoo_adapter import create_pettingzoo_adapter
except ImportError:
    logging.warning("无法导入PettingZoo适配器，PettingZoo环境将不可用")
    create_pettingzoo_adapter = None


class EnvWrapper:
    """VDN统一的环境包装器，兼容所有支持的CTDE环境"""

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

        # 记录环境信息
        logging.info(f"VDN环境已创建: {self.env_name} ({self.difficulty})")
        logging.info(f"智能体数量: {self.n_agents}")
        logging.info(f"智能体ID: {self.agent_ids}")

    def _create_env(self):
        """根据配置创建环境"""
        try:
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
                # PettingZoo环境
                if create_pettingzoo_adapter is None:
                    raise ImportError("PettingZoo适配器不可用，无法使用PettingZoo环境")

                logging.getLogger('pettingzoo').setLevel(logging.WARNING)
                return create_pettingzoo_adapter(
                    env_name=self.env_name,
                    difficulty=self.difficulty,
                    global_state_type=self.global_state_type
                )
            else:
                raise ValueError(f"Unsupported environment: {self.env_name}")

        except ImportError as e:
            logging.error(f"无法导入环境 {self.env_name}: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"创建环境 {self.env_name} 时发生错误: {str(e)}")
            raise

    def reset(self) -> Tuple[Dict[str, np.ndarray], Optional[Dict[str, Any]]]:
        """
        重置环境

        Returns:
            observations: 智能体观测字典
            info: 环境信息
        """
        try:
            obs = self.env.reset()

            # 处理不同环境的返回格式
            if isinstance(obs, tuple):
                obs = obs[0]  # 取观测部分

            # 确保返回字典格式
            if not isinstance(obs, dict):
                obs = {agent_id: obs[i] for i, agent_id in enumerate(self.agent_ids)}

            # 添加全局状态信息
            info = {
                'global_state': self.get_global_state(),
                'episode_step': 0
            }

            return obs, info

        except Exception as e:
            logging.error(f"重置环境时发生错误: {str(e)}")
            raise

    def step(self, actions: Dict[str, int]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, Any]]:
        """
        执行一步

        Args:
            actions: 智能体动作字典

        Returns:
            observations: 下一观测
            rewards: 奖励
            dones: 完成标志
            infos: 环境信息
        """
        try:
            # 将actions转换为环境需要的格式
            if hasattr(self.env, 'step'):
                step_result = self.env.step(actions)
            else:
                raise RuntimeError("Environment does not have step method")

            # 处理不同环境的返回格式
            if len(step_result) == 4:
                obs, rewards, dones, infos = step_result
                truncated = {agent_id: False for agent_id in self.agent_ids}
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

            # 添加episode step信息
            if 'episode_step' not in infos:
                infos['episode_step'] = infos.get('step', 0)

            return obs, rewards, dones, infos

        except Exception as e:
            logging.error(f"执行环境步骤时发生错误: {str(e)}")
            raise

    def get_global_state(self) -> np.ndarray:
        """
        获取全局状态

        Returns:
            全局状态numpy数组
        """
        try:
            if hasattr(self.env, 'get_global_state'):
                return self.env.get_global_state()
            else:
                # 如果环境不支持全局状态，返回所有智能体观测的拼接
                obs = self.env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]

                if not isinstance(obs, dict):
                    obs = {agent_id: obs[i] for i, agent_id in enumerate(self.agent_ids)}

                # 拼接所有智能体的观测
                global_state = np.concatenate([obs[agent_id] for agent_id in self.agent_ids])
                return global_state

        except Exception as e:
            logging.error(f"获取全局状态时发生错误: {str(e)}")
            # 返回零向量作为fallback
            return np.zeros(self.get_env_info()['global_state_dim'])

    def get_avail_actions(self, agent_id: str):
        """
        获取智能体可用动作（如果环境支持）

        Args:
            agent_id: 智能体ID

        Returns:
            可用动作列表或None
        """
        try:
            if hasattr(self.env, 'get_avail_actions'):
                return self.env.get_avail_actions(agent_id)
            return None
        except Exception as e:
            logging.warning(f"获取可用动作时发生错误: {str(e)}")
            return None

    def get_env_info(self) -> Dict[str, Any]:
        """
        获取环境信息

        Returns:
            环境信息字典
        """
        try:
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

            # 添加VDN特定的信息
            info['environment_name'] = self.env_name
            info['difficulty'] = self.difficulty
            info['global_state_type'] = self.global_state_type

            return info

        except Exception as e:
            logging.error(f"获取环境信息时发生错误: {str(e)}")
            raise

    def render(self, mode='human'):
        """
        渲染环境（如果支持）

        Args:
            mode: 渲染模式
        """
        try:
            if hasattr(self.env, 'render'):
                return self.env.render(mode=mode)
            else:
                logging.warning("环境不支持渲染")
                return None
        except Exception as e:
            logging.warning(f"渲染环境时发生错误: {str(e)}")
            return None

    def close(self):
        """关闭环境"""
        try:
            if hasattr(self.env, 'close'):
                self.env.close()
                logging.debug("环境已关闭")
        except Exception as e:
            logging.warning(f"关闭环境时发生错误: {str(e)}")


def create_env_wrapper(config: Dict[str, Any]) -> EnvWrapper:
    """
    创建VDN环境包装器的工厂函数

    Args:
        config: 配置字典

    Returns:
        环境包装器实例
    """
    try:
        return EnvWrapper(config)
    except Exception as e:
        logging.error(f"创建环境包装器时发生错误: {str(e)}")
        raise


def validate_environment_config(config: Dict[str, Any]) -> bool:
    """
    验证环境配置

    Args:
        config: 环境配置

    Returns:
        是否有效
    """
    try:
        # 检查必需的键
        if 'env' not in config:
            logging.error("配置缺少'env'节")
            return False

        env_config = config['env']
        if 'name' not in env_config:
            logging.error("环境配置缺少'name'字段")
            return False

        if 'difficulty' not in env_config:
            logging.error("环境配置缺少'difficulty'字段")
            return False

        # 检查支持的环境
        supported_envs = ['DEM', 'HRG', 'MSFS', 'CM', 'SMAC', 'multiwalker', 'simple_spread', 'simple_crypto']
        if env_config['name'] not in supported_envs:
            logging.error(f"不支持的环境: {env_config['name']}")
            return False

        logging.info("环境配置验证通过")
        return True

    except Exception as e:
        logging.error(f"验证环境配置时发生错误: {str(e)}")
        return False


def get_environment_list() -> List[str]:
    """
    获取支持的环境列表

    Returns:
        支持的环境名称列表
    """
    return ['DEM', 'HRG', 'MSFS', 'CM', 'SMAC', 'multiwalker', 'simple_spread', 'simple_crypto']


def get_difficulty_list(env_name: str) -> List[str]:
    """
    获取指定环境的难度列表

    Args:
        env_name: 环境名称

    Returns:
        难度级别列表
    """
    difficulty_map = {
        'DEM': ['debug', 'easy', 'normal', 'hard'],
        'HRG': ['debug', 'easy_ctde', 'medium_ctde', 'hard_ctde'],
        'MSFS': ['debug', 'easy', 'normal', 'hard'],
        'CM': ['debug', 'easy', 'normal', 'hard'],
        'SMAC': ['debug', 'easy', 'normal', 'hard', '8m', '3s5z', 'MMM'],
        'multiwalker': ['default'],
        'simple_spread': ['default'],
        'simple_crypto': ['default']
    }

    return difficulty_map.get(env_name, ['default'])