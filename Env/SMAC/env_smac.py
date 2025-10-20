"""
SMAC Environment Wrapper

A wrapper for the SMAC library's StarCraft2Env that provides compatibility
with the unified environment interface used by DEM, HRG, and MSFS environments.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from gymnasium import spaces

# Import SMAC library
try:
    from smac.env import StarCraft2Env
except ImportError:
    raise ImportError(
        "SMAC library not found. Please install it with: "
        "pip install smac"
    )

from .config import SMACConfig

# Configure logging
logger = logging.getLogger(__name__)


class SMACEnv:
    """
    SMAC Environment Wrapper

    Wraps the SMAC library's StarCraft2Env to provide compatibility
    with the unified environment interface.
    """

    def __init__(self, config: SMACConfig = None):
        """
        Initialize SMAC environment wrapper

        Args:
            config: Environment configuration
        """
        self.config = config if config is not None else SMACConfig()

        # Initialize SMAC environment
        self.smac_env = StarCraft2Env(
            map_name=self.config.map_name,
            seed=self.config.seed
        )

        # Get environment info
        self.env_info = self.smac_env.get_env_info()

        # Agent IDs (create consistent IDs)
        self.n_agents = self.env_info["n_agents"]
        self.agent_ids = [f"agent_{i}" for i in range(self.n_agents)]

        # Initialize observation and action spaces
        self._setup_spaces()

        # Episode tracking
        self.episode_count = 0
        self.step_count = 0

        logger.info(f"SMAC environment wrapper initialized: {self.config.map_name} "
                   f"({self.n_agents} agents)")

    def _setup_spaces(self) -> None:
        """Setup observation and action spaces"""
        # Action space: discrete actions for all agents
        self.action_spaces = {
            agent_id: spaces.Discrete(self.env_info["n_actions"])
            for agent_id in range(self.env_info["n_agents"])
        }

        # Observation space: based on SMAC observation shape
        obs_shape = self.env_info["obs_shape"]
        self.observation_spaces = {
            agent_id: spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(obs_shape,),
                dtype=np.float32
            ) for agent_id in range(self.env_info["n_agents"])
        }

        # Dimensions for compatibility
        self.act_dims = {agent_id: self.env_info["n_actions"] for agent_id in self.agent_ids}
        self.obs_dims = {agent_id: obs_shape for agent_id in self.agent_ids}

    def reset(self) -> Dict[str, np.ndarray]:
        """
        Reset environment and return initial observations

        Returns:
            Dict[str, np.ndarray]: Initial observations for each agent
        """
        # Reset SMAC environment
        self.smac_env.reset()

        # Reset episode tracking
        self.episode_count += 1
        self.step_count = 0

        # Get observations in unified format
        observations = self.get_observations()

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
        self.step_count += 1

        # Convert actions to SMAC format (list)
        smac_actions = []
        for i in range(self.n_agents):
            agent_id = f"agent_{i}"
            if agent_id in actions:
                action = int(actions[agent_id])
            else:
                # Default action if not provided
                avail_actions = self.smac_env.get_avail_agent_actions(i)
                avail_actions_ind = np.nonzero(avail_actions)[0]
                action = np.random.choice(avail_actions_ind) if len(avail_actions_ind) > 0 else 0

            smac_actions.append(action)

        # Execute step in SMAC environment
        reward, terminated, info = self.smac_env.step(smac_actions)

        # Get observations in unified format
        observations = self.get_observations()

        # Create rewards dictionary (SMAC returns single reward)
        rewards = {agent_id: float(reward / self.n_agents) for agent_id in self.agent_ids}

        # Create dones dictionary
        dones = {agent_id: terminated for agent_id in self.agent_ids}

        # Create info dictionary
        infos = {
            agent_id: {
                'episode_step': self.step_count,
                'max_steps': self.env_info['episode_limit']
            } for agent_id in self.agent_ids
        }
        infos['episode'] = {
            'step': self.step_count,
            'max_steps': self.env_info['episode_limit'],
            'terminated': terminated
        }

        return observations, rewards, dones, infos

    def get_observations(self) -> Dict[str, np.ndarray]:
        """Get observations for all agents in unified format"""
        smac_obs = self.smac_env.get_obs()
        observations = {}

        for i, obs in enumerate(smac_obs):
            agent_id = f"agent_{i}"
            observations[agent_id] = obs.astype(np.float32)

        return observations

    def get_global_state(self) -> np.ndarray:
        """Get global state representation (for CTDE algorithms)"""
        return self.smac_env.get_state()

    def get_avail_actions(self, agent_id: str) -> List[int]:
        """Get available actions for an agent (for action masking)"""
        if not agent_id.startswith("agent_"):
            raise ValueError(f"Invalid agent ID: {agent_id}")

        agent_idx = int(agent_id.split("_")[1])
        if agent_idx >= self.n_agents:
            return []

        avail_actions = self.smac_env.get_avail_agent_actions(agent_idx)
        avail_actions_ind = np.nonzero(avail_actions)[0].tolist()

        return avail_actions_ind

    def get_env_info(self) -> Dict[str, Any]:
        """Get environment information"""
        info = self.env_info.copy()
        info.update({
            'n_agents': self.n_agents,
            'agent_ids': self.agent_ids,
            'action_spaces': self.action_spaces,
            'observation_spaces': self.observation_spaces,
            'act_dims': self.act_dims,
            'obs_dims': self.obs_dims,
            'max_steps': self.env_info['episode_limit'],
            'episode_limit': self.env_info['episode_limit'],
            'n_actions': self.env_info['n_actions'],
            'obs_shape': self.env_info['obs_shape'],
            'state_shape': self.env_info['state_shape'],
            'global_state_dim': self.env_info['state_shape']
        })
        return info

    def render(self, mode='human'):
        """Render the environment (not supported in SMAC wrapper)"""
        logger.warning("Rendering not supported in SMAC wrapper")
        return None

    def close(self):
        """Close the environment"""
        self.smac_env.close()

    def seed(self, seed: int):
        """Set random seed"""
        # Note: SMAC environment doesn't support re-seeding after initialization
        logger.warning(f"SMAC environment doesn't support re-seeding. Use seed in constructor instead.")

    # SMAC-specific methods
    def get_obs(self) -> List[np.ndarray]:
        """Get observations in SMAC format (list)"""
        return self.smac_env.get_obs()

    def get_state(self) -> np.ndarray:
        """Get state in SMAC format"""
        return self.smac_env.get_state()

    def get_avail_agent_actions(self, agent_idx: int) -> np.ndarray:
        """Get available actions for agent in SMAC format"""
        return self.smac_env.get_avail_agent_actions(agent_idx)


# Factory function for easy environment creation
def create_smac_env(map_name: str = "8m", config: SMACConfig = None, **kwargs) -> SMACEnv:
    """
    Create SMAC environment with specified map

    Args:
        map_name: SMAC map name ("8m", "3s", "2s3z", "1c3s5z", etc.)
        config: Pre-configured SMACConfig object (overrides map_name and kwargs)
        **kwargs: Additional configuration parameters

    Returns:
        SMACEnv: Configured environment
    """
    if config is not None:
        return SMACEnv(config)
    else:
        config = SMACConfig(map_name=map_name, **kwargs)
        return SMACEnv(config)


def create_smac_env_easy(**kwargs) -> SMACEnv:
    """Create easy difficulty SMAC environment"""
    from .config import get_easy_config
    config = get_easy_config()
    for key, value in kwargs.items():
        setattr(config, key, value)
    return SMACEnv(config)


def create_smac_env_normal(**kwargs) -> SMACEnv:
    """Create normal difficulty SMAC environment"""
    from .config import get_normal_config
    config = get_normal_config()
    for key, value in kwargs.items():
        setattr(config, key, value)
    return SMACEnv(config)


def create_smac_env_hard(**kwargs) -> SMACEnv:
    """Create hard difficulty SMAC environment"""
    from .config import get_hard_config
    config = get_hard_config()
    for key, value in kwargs.items():
        setattr(config, key, value)
    return SMACEnv(config)