"""
HRG (Heterogeneous Resource Gathering) Environment - Ultra Fast CTDE Version

Ultra-fast CTDE (Centralized Training with Decentralized Execution) implementation
for extreme training speed with minimal agent count.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from gymnasium import spaces

from .env_hrg_ultra_fast import HRGUltraFastEnv, UltraFastConfig
from .core import AgentType, ActionType

logger = logging.getLogger(__name__)


class HRGUltraFastCTDEEnv(HRGUltraFastEnv):
    """Ultra-fast CTDE environment wrapper"""

    def __init__(self, config: UltraFastConfig = None):
        # Global state dimension: 2 agents * 8 dims + 6 resource dims = 22
        self.global_state_dim = 22

        super().__init__(config)

        # Override observation spaces for CTDE
        self._setup_ctde_spaces()

    def _setup_ctde_spaces(self):
        """Setup CTDE-specific spaces"""
        # Individual observations remain the same
        self.observation_spaces = {
            agent_id: spaces.Box(low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32)
            for agent_id in self.agent_ids
        }

        # Add global state space
        self.global_state_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.global_state_dim,), dtype=np.float32
        )

    def reset(self, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset environment and return initial observations and info"""
        observations = super().reset()

        # Create info dict with global state
        info = {
            'global_state': self.get_global_state(),
            'episode': {
                'step': 0,
                'total_score': self.game_state.total_score
            }
        }

        return observations, info

    def step(self, actions: Dict[str, Union[int, np.ndarray]]) -> Tuple[
        Dict[str, np.ndarray], Dict[str, float], bool, bool, Dict[str, Any]
    ]:
        """Execute step with CTDE format"""
        observations, rewards, dones, infos = super().step(actions)

        # Add global state to info
        all_done = all(dones.values())
        infos['global_state'] = self.get_global_state()
        infos['episode'] = infos.get('episode', {})
        infos['episode']['step'] = self.step_count

        return observations, rewards, all_done, False, infos

    def get_env_info(self) -> Dict[str, Any]:
        """Get environment information for algorithms"""
        return {
            'n_agents': self.n_agents,
            'agent_ids': self.agent_ids,
            'obs_dims': self.obs_dims,
            'act_dims': self.act_dims,
            'global_state_dim': self.global_state_dim,
            'episode_limit': self.config.max_steps,
            'action_space': list(self.action_spaces.values())[0],
            'observation_space': list(self.observation_spaces.values())[0]
        }


# Factory function
def create_hrg_ultra_fast_ctde_env(**kwargs) -> HRGUltraFastCTDEEnv:
    """Create ultra-fast CTDE environment"""
    config = UltraFastConfig(**kwargs)
    return HRGUltraFastCTDEEnv(config)