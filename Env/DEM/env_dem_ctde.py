"""
DEM Environment CTDE Wrapper

This module provides a CTDE (Centralized Training, Decentralized Execution)
wrapper for the DEM environment, making it compatible with multi-agent
reinforcement learning algorithms like QMIX, VDN, etc.
"""

import numpy as np
import random
from typing import Dict, List, Any, Tuple, Optional, Union
import logging

from .env_dem import DEMEnv, DEMConfig, create_dem_env

# Configure logging
logger = logging.getLogger(__name__)


class DEMCTDEWrapper(DEMEnv):
    """
    CTDE-compatible wrapper for DEM environment

    This wrapper extends the base DEM environment to provide CTDE-specific
    functionality including global state representation and standardized
    interfaces for centralized training algorithms.
    """

    def __init__(self,
                 difficulty: str = "normal",
                 global_state_type: str = "concat",
                 global_state_dim: Optional[int] = None,
                 **kwargs):
        """
        Initialize CTDE wrapper for DEM environment

        Args:
            difficulty: Difficulty level ("easy", "normal", "hard")
            global_state_type: Type of global state representation
                ("concat", "mean", "max", "attention")
            global_state_dim: Dimension of global state (if None, auto-calculated)
            **kwargs: Additional environment configuration
        """
        # Set CTDE-specific settings before calling parent __init__
        self.global_state_type = global_state_type
        self._global_state_dim = global_state_dim

        # Validate global state type
        valid_types = ["concat", "mean", "max", "attention"]
        if self.global_state_type not in valid_types:
            raise ValueError(f"global_state_type must be one of {valid_types}, "
                           f"got {self.global_state_type}")

        # Initialize base environment
        config = DEMConfig(difficulty=difficulty, **kwargs)
        super().__init__(config)

        # Calculate global state dimension if not provided
        if self._global_state_dim is None:
            self._global_state_dim = self._calculate_global_state_dim()

        logger.info(f"DEM CTDE Wrapper initialized with global_state_type='{global_state_type}', "
                   f"global_state_dim={self._global_state_dim}")

    def _calculate_global_state_dim(self) -> int:
        """Calculate global state dimension based on type"""
        if self.global_state_type == "concat":
            return super().get_global_state().shape[0]
        elif self.global_state_type == "mean":
            return self.observation_space.shape[0]  # Same as observation dim
        elif self.global_state_type == "max":
            return self.observation_space.shape[0]  # Same as observation dim
        elif self.global_state_type == "attention":
            # Attention-based state: obs + attention weights
            return self.observation_space.shape[0] + self.config.num_agents
        else:
            return super().get_global_state().shape[0]

    def get_global_state(self) -> np.ndarray:
        """
        Get global state representation for centralized training

        Returns:
            Global state vector
        """
        if self.global_state_type == "concat":
            return self._get_concat_global_state()
        elif self.global_state_type == "mean":
            return self._get_mean_global_state()
        elif self.global_state_type == "max":
            return self._get_max_global_state()
        elif self.global_state_type == "attention":
            return self._get_attention_global_state()
        else:
            return self._get_concat_global_state()

    def _get_concat_global_state(self) -> np.ndarray:
        """Get concatenated global state"""
        return super().get_global_state()

    def _get_mean_global_state(self) -> np.ndarray:
        """Get mean-pooled global state from agent observations"""
        observations = self.get_observations()
        if not observations:
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)

        # Stack all observations and take mean
        obs_array = np.stack(list(observations.values()))
        mean_obs = np.mean(obs_array, axis=0)

        return mean_obs

    def _get_max_global_state(self) -> np.ndarray:
        """Get max-pooled global state from agent observations"""
        observations = self.get_observations()
        if not observations:
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)

        # Stack all observations and take max
        obs_array = np.stack(list(observations.values()))
        max_obs = np.max(obs_array, axis=0)

        return max_obs

    def _get_attention_global_state(self) -> np.ndarray:
        """Get attention-based global state"""
        observations = self.get_observations()
        if not observations:
            return np.zeros(self.observation_space.shape[0] + self.config.num_agents, dtype=np.float32)

        # Stack all observations and take mean
        obs_array = np.stack(list(observations.values()))
        mean_obs = np.mean(obs_array, axis=0)

        # Add attention weights (uniform for now, could be learned)
        attention_weights = np.ones(self.config.num_agents) / self.config.num_agents

        return np.concatenate([mean_obs, attention_weights])

    def get_env_info(self) -> Dict[str, Any]:
        """
        Get environment information for CTDE algorithms

        Returns:
            Environment information dictionary
        """
        base_info = super().get_env_info()

        # Add CTDE-specific information
        ctde_info = {
            'global_state_type': self.global_state_type,
            'global_state_dim': self._global_state_dim,
            'centralized_training': True,
            'decentralized_execution': True,
        }

        base_info.update(ctde_info)
        return base_info

    def get_agent_obs(self, agent_id: str) -> np.ndarray:
        """
        Get observation for a specific agent

        Args:
            agent_id: ID of the agent

        Returns:
            Agent observation vector
        """
        observations = self.get_observations()
        return observations.get(agent_id, np.zeros(self.observation_space.shape[0], dtype=np.float32))

    def get_agent_actions_mask(self, agent_id: str) -> np.ndarray:
        """
        Get action mask for a specific agent

        Args:
            agent_id: ID of the agent

        Returns:
            Boolean array indicating valid actions
        """
        if agent_id not in self.game_state.agents:
            return np.ones(self.action_space.n, dtype=bool)

        agent = self.game_state.agents[agent_id]
        if not agent.is_alive:
            return np.ones(self.action_space.n, dtype=bool)

        mask = np.ones(self.action_space.n, dtype=bool)

        # All agents can always stay and observe
        # All agents can always move (terrain check handled in environment)
        # Attack action validity checked in environment
        # Communication actions always valid
        # Guard action validity checked in environment

        return mask

    def get_avail_agent_actions(self, agent_id: str) -> List[int]:
        """
        Get available actions for a specific agent

        Args:
            agent_id: ID of the agent

        Returns:
            List of available action indices
        """
        mask = self.get_agent_actions_mask(agent_id)
        return [i for i, available in enumerate(mask) if available]

    def get_stats(self) -> Dict[str, Any]:
        """
        Get detailed episode statistics

        Returns:
            Statistics dictionary
        """
        stats = self.game_state.stats.copy()
        stats.update({
            'episode_step': self.game_state.current_step,
            'max_steps': self.game_state.max_steps,
            'vip_hp': self.game_state.vip.hp,
            'vip_max_hp': self.game_state.vip.max_hp,
            'vip_pos': self.game_state.vip.pos,
            'vip_target_pos': self.game_state.vip.target_pos,
            'vip_distance_to_target': self.game_state.stats['vip_distance_to_target'],
            'threats_alive': len([t for t in self.game_state.threats.values() if t.is_alive]),
            'agents_alive': len([a for a in self.game_state.agents.values() if a.is_alive]),
            'total_agents': self.config.num_agents,
            'messages_sent': self.game_state.stats['messages_sent'],
            'threats_killed': self.game_state.stats['threats_killed'],
            'agents_killed': self.game_state.stats['agents_killed'],
            'body_blocks': self.game_state.stats['body_blocks'],
            'long_range_kills': self.game_state.stats['long_range_kills'],
            'termination_reason': self.game_state.termination_reason if self.game_state.is_terminated else None,
            'is_terminated': self.game_state.is_terminated,
            'total_reward': self.game_state.total_reward,
        })

        # Add role-specific statistics
        stats.update({
            'agents_adjacent_to_vip': self.game_state.stats['agents_adjacent_to_vip'],
            'agents_ahead_of_vip': self.game_state.stats['agents_ahead_of_vip'],
            'agent_spread': self.game_state.stats['agent_spread'],
        })

        return stats

    def get_agent_positions(self) -> Dict[str, Tuple[int, int]]:
        """
        Get current positions of all agents

        Returns:
            Dictionary mapping agent IDs to positions
        """
        positions = {}
        for agent_id, agent in self.game_state.agents.items():
            if agent.is_alive:
                positions[agent_id] = (agent.pos.x, agent.pos.y)
        return positions

    def get_threat_positions(self) -> List[Tuple[int, int, str]]:
        """
        Get current positions and types of all threats

        Returns:
            List of tuples (x, y, threat_type)
        """
        positions = []
        for threat in self.game_state.threats.values():
            if threat.is_alive:
                positions.append((threat.pos.x, threat.pos.y, threat.type.value))
        return positions

    def get_global_info(self) -> Dict[str, Any]:
        """
        Get global information for visualization and debugging

        Returns:
            Global information dictionary
        """
        return {
            'vip': {
                'pos': (self.game_state.vip.pos.x, self.game_state.vip.pos.y),
                'target_pos': (self.game_state.vip.target_pos.x, self.game_state.vip.target_pos.y),
                'hp': self.game_state.vip.hp,
                'max_hp': self.game_state.vip.max_hp,
                'is_alive': self.game_state.vip.is_alive,
                'is_under_attack': self.game_state.vip.is_under_attack,
            },
            'agents': self.get_agent_positions(),
            'threats': self.get_threat_positions(),
            'terrain': self._get_terrain_map(),
            'step': self.game_state.current_step,
            'max_steps': self.game_state.max_steps,
            'messages': self.game_state.messages[-5:],  # Last 5 messages
            'stats': self.get_stats(),
        }

    def _get_terrain_map(self) -> List[List[str]]:
        """
        Get terrain map as list of lists

        Returns:
            2D list of terrain types
        """
        terrain_map = []
        for x in range(self.config.grid_size):
            row = []
            for y in range(self.config.grid_size):
                terrain = self.game_state.terrain[x, y]
                row.append(terrain.value)
            terrain_map.append(row)
        return terrain_map

    def save_replay(self, filepath: str) -> None:
        """
        Save episode replay (placeholder for future implementation)

        Args:
            filepath: Path to save replay file
        """
        logger.info(f"Replay saving not yet implemented for DEM environment")
        pass

    def load_replay(self, filepath: str) -> None:
        """
        Load episode replay (placeholder for future implementation)

        Args:
            filepath: Path to replay file
        """
        logger.info(f"Replay loading not yet implemented for DEM environment")
        pass

    def seed(self, seed: int) -> None:
        """
        Set random seed for reproducibility

        Args:
            seed: Random seed
        """
        self.config.seed = seed
        np.random.seed(seed)
        random.seed(seed)

        # Reinitialize with new seed
        self._apply_config_to_state()

        logger.info(f"Environment seeded with {seed}")


def create_dem_ctde_env(difficulty: str = "normal",
                        global_state_type: str = "concat",
                        **kwargs) -> DEMCTDEWrapper:
    """
    Create a DEM CTDE environment with specified parameters

    Args:
        difficulty: Difficulty level ("easy", "normal", "hard")
        global_state_type: Type of global state representation
        **kwargs: Additional configuration parameters

    Returns:
        DEM CTDE environment instance
    """
    return DEMCTDEWrapper(
        difficulty=difficulty,
        global_state_type=global_state_type,
        **kwargs
    )


# Predefined CTDE environment configurations
def create_dem_ctde_env_easy(**kwargs) -> DEMCTDEWrapper:
    """Create easy difficulty DEM CTDE environment"""
    return create_dem_ctde_env(difficulty="easy", **kwargs)


def create_dem_ctde_env_normal(**kwargs) -> DEMCTDEWrapper:
    """Create normal difficulty DEM CTDE environment"""
    return create_dem_ctde_env(difficulty="normal", **kwargs)


def create_dem_ctde_env_hard(**kwargs) -> DEMCTDEWrapper:
    """Create hard difficulty DEM CTDE environment"""
    return create_dem_ctde_env(difficulty="hard", **kwargs)


# Convenience functions for different global state types
def create_dem_ctde_env_concat(difficulty: str = "normal", **kwargs) -> DEMCTDEWrapper:
    """Create DEM CTDE environment with concatenated global state"""
    return create_dem_ctde_env(difficulty=difficulty, global_state_type="concat", **kwargs)


def create_dem_ctde_env_mean(difficulty: str = "normal", **kwargs) -> DEMCTDEWrapper:
    """Create DEM CTDE environment with mean-pooled global state"""
    return create_dem_ctde_env(difficulty=difficulty, global_state_type="mean", **kwargs)


def create_dem_ctde_env_max(difficulty: str = "normal", **kwargs) -> DEMCTDEWrapper:
    """Create DEM CTDE environment with max-pooled global state"""
    return create_dem_ctde_env(difficulty=difficulty, global_state_type="max", **kwargs)


def create_dem_ctde_env_attention(difficulty: str = "normal", **kwargs) -> DEMCTDEWrapper:
    """Create DEM CTDE environment with attention-based global state"""
    return create_dem_ctde_env(difficulty=difficulty, global_state_type="attention", **kwargs)