"""
HRG Fast CTDE (Centralized Training Decentralized Execution) Wrapper

Fast training version for CTDE algorithms like QMIX, VDN, etc.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union

from .env_hrg_fast import HRGFastEnv, HRGFastConfig


class HRGFastCTDEEnv:
    """Fast CTDE wrapper for HRG environment"""

    def __init__(self, env: HRGFastEnv = None, config: HRGFastConfig = None):
        if env is not None:
            self.env = env
        elif config is not None:
            self.env = HRGFastEnv(config)
        else:
            self.env = HRGFastEnv(HRGFastConfig())

        self.n_agents = self.env.n_agents
        self.agent_ids = self.env.agent_ids

        # Calculate dynamic global state dimension
        # Each agent: 8 dims, plus 6 dims for resource summary
        global_state_dim = self.n_agents * 8 + 6

        # Environment info
        self.env_info = {
            'n_agents': self.n_agents,
            'agent_ids': self.agent_ids,
            'obs_dims': self.env.obs_dims,
            'act_dims': self.env.act_dims,
            'global_state_dim': global_state_dim
        }

    def reset(self) -> Dict[str, np.ndarray]:
        """Reset environment"""
        return self.env.reset()

    def step(self, actions: Dict[str, Union[int, np.ndarray]]) -> Tuple[
        Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, Any]
    ]:
        """Execute step and add global state to infos"""
        observations, rewards, dones, infos = self.env.step(actions)
        
        # Add global state
        infos['global_state'] = self.get_global_state()
        
        return observations, rewards, dones, infos

    def get_global_state(self) -> np.ndarray:
        """Get global state"""
        return self.env.get_global_state()

    def get_env_info(self) -> Dict[str, Any]:
        """Get environment info"""
        return self.env_info.copy()

    def get_avail_actions(self, agent_id: str) -> List[int]:
        """Get available actions"""
        return self.env.get_avail_actions(agent_id)

    def close(self):
        """Close environment"""
        self.env.close()

    def seed(self, seed: int):
        """Set random seed"""
        np.random.seed(seed)


# Factory functions
def create_hrg_fast_ctde_env(difficulty: str = "fast_training", **kwargs) -> HRGFastCTDEEnv:
    """Create fast CTDE environment"""
    from .env_hrg_fast import create_hrg_fast_env
    base_env = create_hrg_fast_env(difficulty, **kwargs)
    return HRGFastCTDEEnv(base_env)


def create_hrg_ultra_fast_ctde_env(**kwargs) -> HRGFastCTDEEnv:
    """Create ultra-fast CTDE environment for rapid training"""
    from .env_hrg_fast import create_hrg_fast_env
    base_env = create_hrg_fast_env("ultra_fast", **kwargs)
    return HRGFastCTDEEnv(base_env)
