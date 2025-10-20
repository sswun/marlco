"""
SMAC CTDE (Centralized Training Decentralized Execution) Wrapper

Provides a wrapper for SMAC environment that supports centralized training
with decentralized execution, compatible with algorithms like QMIX, VDN, etc.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union

from .env_smac import SMACEnv
from .config import SMACConfig

logger = logging.getLogger(__name__)


class SMACCTDEEnv:
    """
    SMAC CTDE Environment Wrapper

    Wraps the SMAC environment to provide centralized training capabilities
    while maintaining decentralized execution using the original SMAC global state.
    """

    def __init__(self, env: SMACEnv = None, config: SMACConfig = None):
        """
        Initialize SMAC CTDE environment

        Args:
            env: Base SMAC environment (if None, creates one from config)
            config: Configuration for creating environment (used only if env is None)
        """
        if env is not None:
            self.env = env
        elif config is not None:
            self.env = SMACEnv(config)
        else:
            # Default configuration
            self.env = SMACEnv(SMACConfig())

        self.n_agents = self.env.n_agents
        self.agent_ids = self.env.agent_ids

        # Environment info (for compatibility)
        self.env_info = self.env.get_env_info()

        # Use original SMAC global state dimension
        self.global_state_dim = self.env_info['state_shape']
        self.env_info['global_state_dim'] = self.global_state_dim

        logger.info(f"SMAC CTDE environment initialized with {self.n_agents} agents "
                   f"using original global state (dim={self.global_state_dim})")

    def reset(self) -> Dict[str, np.ndarray]:
        """
        Reset environment and return initial observations

        Returns:
            Dict[str, np.ndarray]: Initial observations for each agent
        """
        observations = self.env.reset()
        return observations

    def step(self, actions: Dict[str, Union[int, np.ndarray]]) -> Tuple[
        Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, Any]
    ]:
        """
        Execute one step in the environment

        Args:
            actions: Dictionary mapping agent IDs to actions

        Returns:
            Tuple of (observations, rewards, dones, infos)
        """
        observations, rewards, dones, infos = self.env.step(actions)

        # Add global state to infos for CTDE algorithms
        global_state = self.get_global_state()
        infos['global_state'] = global_state

        return observations, rewards, dones, infos

    def get_global_state(self) -> np.ndarray:
        """
        Get global state representation for centralized training
        Uses the original SMAC global state directly

        Returns:
            np.ndarray: Global state representation from original SMAC environment
        """
        # Use the original SMAC global state directly
        return self.env.get_global_state().astype(np.float32)

    def get_env_info(self) -> Dict[str, Any]:
        """Get environment information"""
        info = self.env_info.copy()
        info['global_state_dim'] = self.global_state_dim
        return info

    def get_avail_actions(self, agent_id: str) -> List[int]:
        """Get available actions for an agent"""
        return self.env.get_avail_actions(agent_id)

    def get_avail_agent_actions(self, agent_id: str) -> List[int]:
        """Alias for get_avail_actions (compatibility with SMAC)"""
        return self.get_avail_actions(agent_id)

    def get_obs(self) -> List[np.ndarray]:
        """Get observations as list (compatibility with SMAC)"""
        return self.env.get_obs()

    def get_state(self) -> np.ndarray:
        """Get global state (compatibility with SMAC)"""
        return self.get_global_state()

    def render(self, mode='human'):
        """Render the environment"""
        return self.env.render(mode=mode)

    def close(self):
        """Close the environment"""
        self.env.close()

    def seed(self, seed: int):
        """Set random seed"""
        return self.env.seed(seed)

    def __getattr__(self, name):
        """Delegate attribute access to underlying environment"""
        return getattr(self.env, name)


# Factory functions for creating CTDE environments
def create_smac_ctde_env(map_name: str = "8m", **kwargs) -> SMACCTDEEnv:
    """
    Create SMAC CTDE environment with specified map

    Args:
        map_name: SMAC map name ("8m", "3s", "2s3z", "1c3s5z", etc.)
        **kwargs: Additional configuration parameters

    Returns:
        SMACCTDEEnv: Configured CTDE environment
    """
    from .config import SMACConfig
    config = SMACConfig(map_name=map_name, **kwargs)
    base_env = SMACEnv(config)
    return SMACCTDEEnv(base_env)


def create_smac_ctde_env_easy(**kwargs) -> SMACCTDEEnv:
    """Create easy difficulty SMAC CTDE environment"""
    from .config import get_easy_config
    config = get_easy_config()
    for key, value in kwargs.items():
        setattr(config, key, value)
    base_env = SMACEnv(config)
    return SMACCTDEEnv(base_env)


def create_smac_ctde_env_normal(**kwargs) -> SMACCTDEEnv:
    """Create normal difficulty SMAC CTDE environment"""
    from .config import get_normal_config
    config = get_normal_config()
    for key, value in kwargs.items():
        setattr(config, key, value)
    base_env = SMACEnv(config)
    return SMACCTDEEnv(base_env)


def create_smac_ctde_env_hard(**kwargs) -> SMACCTDEEnv:
    """Create hard difficulty SMAC CTDE environment"""
    from .config import get_hard_config
    config = get_hard_config()
    for key, value in kwargs.items():
        setattr(config, key, value)
    base_env = SMACEnv(config)
    return SMACCTDEEnv(base_env)


# Predefined environment configurations for CTDE
def create_smac_debug_env(**kwargs) -> SMACCTDEEnv:
    """Create debug SMAC CTDE environment"""
    from .config import get_debug_config
    config = get_debug_config()
    for key, value in kwargs.items():
        setattr(config, key, value)
    base_env = SMACEnv(config)
    return SMACCTDEEnv(base_env)


def create_smac_corridor_env(**kwargs) -> SMACCTDEEnv:
    """Create corridor SMAC CTDE environment"""
    base_env = create_smac_env(map_name="corridor", **kwargs)
    return SMACCTDEEnv(base_env)


def create_smac_6h_env(**kwargs) -> SMACCTDEEnv:
    """Create 6h SMAC CTDE environment"""
    base_env = create_smac_env(map_name="6h", **kwargs)
    return SMACCTDEEnv(base_env)


def create_smac_mmm_env(**kwargs) -> SMACCTDEEnv:
    """Create MMM SMAC CTDE environment"""
    base_env = create_smac_env(map_name="MMM", **kwargs)
    return SMACCTDEEnv(base_env)