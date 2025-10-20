"""
环境包装器 - 统一CTDE环境接口
"""
import sys
import os
import logging
import numpy as np
from typing import Dict, Any, Tuple, Optional

# 添加Env路径到sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../Env'))

# 导入PettingZoo适配器
from .pettingzoo_adapter import create_pettingzoo_adapter


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
            # PettingZoo环境
            logging.getLogger('pettingzoo').setLevel(logging.WARNING)
            return create_pettingzoo_adapter(
                env_name=self.env_name,
                difficulty=self.difficulty,
                global_state_type=self.global_state_type
            )
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

    def close(self):
        """关闭环境"""
        if hasattr(self.env, 'close'):
            self.env.close()


def create_env_wrapper(config: Dict[str, Any]) -> EnvWrapper:
    """创建环境包装器的工厂函数"""
    return EnvWrapper(config)