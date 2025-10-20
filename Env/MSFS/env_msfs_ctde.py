"""
MSFS Environment CTDE Wrapper

This module provides a Centralized Training, Decentralized Execution (CTDE) wrapper
for the MSFS environment, making it compatible with multi-agent reinforcement learning algorithms.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from .env_msfs import MSFSEnv
from .config import MSFSConfig, MSFSPresetConfigs
from .core import GameState

import logging

logger = logging.getLogger(__name__)


class MSFSCTDEWrapper(MSFSEnv):
    """
    Centralized Training, Decentralized Execution (CTDE) wrapper for MSFS environment

    This wrapper provides centralized global state for training while maintaining
    decentralized execution for agents, making it compatible with algorithms like:
    - QMIX, VDN (value decomposition)
    - MADDPG (centralized critic)
    - MAPPO (centralized critic)
    - COMA (counterfactual baseline)
    """

    def __init__(self, difficulty: str = "normal", global_state_type: str = "concat", **kwargs):
        """
        Initialize MSFS CTDE environment

        Args:
            difficulty: Environment difficulty ("easy", "normal", "hard")
            global_state_type: Type of global state representation ("concat", "mean", "max", "attention")
            **kwargs: Additional configuration parameters
        """
        # Initialize base environment
        config = MSFSPresetConfigs.normal() if difficulty == "normal" else \
                MSFSPresetConfigs.easy() if difficulty == "easy" else \
                MSFSPresetConfigs.hard()

        # Apply overrides
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        super().__init__(config)

        # CTDE specific attributes
        self.global_state_type = global_state_type
        self._global_state_dim = None

        # Validate global state type
        valid_types = ["concat", "mean", "max", "attention"]
        if self.global_state_type not in valid_types:
            raise ValueError(f"global_state_type must be one of {valid_types}, "
                           f"got {self.global_state_type}")

        # Calculate global state dimension if not provided
        if self._global_state_dim is None:
            self._global_state_dim = self._calculate_global_state_dim()

        logger.info(f"MSFS CTDE Wrapper initialized with global_state_type='{global_state_type}', "
                   f"global_state_dim={self._global_state_dim}")

    def _calculate_global_state_dim(self) -> int:
        """Calculate global state dimension based on type"""
        if self.global_state_type == "concat":
            return self.get_global_state().shape[0]
        elif self.global_state_type == "mean":
            return self.observation_space.shape[0]  # Same as observation dim
        elif self.global_state_type == "max":
            return self.observation_space.shape[0]  # Same as observation dim
        elif self.global_state_type == "attention":
            # Attention-based state: obs + attention weights
            return self.observation_space.shape[0] + self.config.num_agents
        else:
            return self.get_global_state().shape[0]

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
            "global_state_dim": self._global_state_dim,
            "global_state_type": self.global_state_type,
            "centralized_training": True,
            "decentralized_execution": True,
            "episode_limit": self.config.max_steps
        }

        # Merge dictionaries
        base_info.update(ctde_info)

        return base_info

    def get_global_info(self) -> Dict[str, Any]:
        """
        Get comprehensive global information for analysis

        Returns:
            Global information dictionary
        """
        observations = self.get_observations()
        global_state = self.get_global_state()

        info = {
            "observations": observations,
            "global_state": global_state,
            "global_state_type": self.global_state_type,
            "game_state": self.game_state,
            "stats": {
                "orders_completed": self.game_state.orders_completed,
                "simple_orders_completed": self.game_state.simple_orders_completed,
                "complex_orders_completed": self.game_state.complex_orders_completed,
                "total_orders_generated": self.game_state.total_orders_generated,
                "specialization_events": self.game_state.specialization_events,
                "total_reward": self.game_state.total_reward,
                "current_step": self.game_state.current_step,
                "max_steps": self.game_state.max_steps
            },
            "queue_stats": self.game_state.get_queue_stats(),
            "role_emergence_stats": self.game_state.get_role_emergence_stats(),
            "agent_states": {}
        }

        # Add individual agent states
        for agent_id, agent in self.game_state.agents.items():
            info["agent_states"][agent_id] = {
                "current_workstation": agent.current_workstation.value,
                "move_cooldown": agent.move_cooldown,
                "carrying_order": agent.carrying_order is not None,
                "specialization_count": agent.specialization_count,
                "consecutive_specialization": agent.consecutive_specialization
            }

        return info

    def get_stats(self) -> Dict[str, Any]:
        """
        Get current statistics for analysis

        Returns:
            Statistics dictionary
        """
        return {
            "orders_completed": self.game_state.orders_completed,
            "simple_orders_completed": self.game_state.simple_orders_completed,
            "complex_orders_completed": self.game_state.complex_orders_completed,
            "total_orders_generated": self.game_state.total_orders_generated,
            "specialization_events": self.game_state.specialization_events,
            "total_reward": self.game_state.total_reward,
            "current_step": self.game_state.current_step,
            "max_steps": self.game_state.max_steps,
            "order_completion_rate": (
                self.game_state.orders_completed / max(1, self.game_state.total_orders_generated)
            ),
            "complex_order_ratio": (
                self.game_state.complex_orders_completed / max(1, self.game_state.orders_completed)
            ),
            "avg_orders_per_step": (
                self.game_state.orders_completed / max(1, self.game_state.current_step)
            )
        }

    def get_agent_observations(self) -> Dict[str, np.ndarray]:
        """
        Get observations for all agents (alias for get_observations)

        Returns:
            Dictionary mapping agent IDs to observations
        """
        return self.get_observations()

    def get_agent_actions_count(self) -> int:
        """
        Get number of possible actions for each agent

        Returns:
            Number of actions
        """
        return self.action_space.n

    def get_agents(self) -> List[str]:
        """
        Get list of agent IDs

        Returns:
            List of agent IDs
        """
        return list(self.game_state.agents.keys())

    def get_observation_shape(self) -> Tuple[int, ...]:
        """
        Get observation shape for each agent

        Returns:
            Observation shape
        """
        return self.observation_space.shape

    def get_global_state_shape(self) -> Tuple[int, ...]:
        """
        Get global state shape

        Returns:
            Global state shape
        """
        return (self._global_state_dim,)

    def is_terminal(self) -> bool:
        """
        Check if the environment is in terminal state

        Returns:
            True if terminal, False otherwise
        """
        return self.game_state.is_terminated()

    def get_won(self) -> bool:
        """
        Check if the episode was won (high completion rate)

        Returns:
            True if won, False otherwise
        """
        if self.game_state.total_orders_generated == 0:
            return False
        completion_rate = self.game_state.orders_completed / self.game_state.total_orders_generated
        return completion_rate >= 0.8  # 80% completion rate is considered winning

    def get_score(self) -> float:
        """
        Get episode score

        Returns:
            Episode score
        """
        return self.game_state.total_reward


def create_msfs_ctde_env(difficulty: str = "normal",
                        global_state_type: str = "concat",
                        **kwargs) -> MSFSCTDEWrapper:
    """
    Create a MSFS CTDE environment with specified parameters

    Args:
        difficulty: Difficulty level ("easy", "normal", "hard")
        global_state_type: Type of global state representation
        **kwargs: Additional configuration parameters

    Returns:
        MSFS CTDE environment instance
    """
    return MSFSCTDEWrapper(
        difficulty=difficulty,
        global_state_type=global_state_type,
        **kwargs
    )


# Predefined CTDE environment configurations
def create_msfs_ctde_env_easy(**kwargs) -> MSFSCTDEWrapper:
    """Create easy difficulty MSFS CTDE environment"""
    return create_msfs_ctde_env(difficulty="easy", **kwargs)


def create_msfs_ctde_env_normal(**kwargs) -> MSFSCTDEWrapper:
    """Create normal difficulty MSFS CTDE environment"""
    return create_msfs_ctde_env(difficulty="normal", **kwargs)


def create_msfs_ctde_env_hard(**kwargs) -> MSFSCTDEWrapper:
    """Create hard difficulty MSFS CTDE environment"""
    return create_msfs_ctde_env(difficulty="hard", **kwargs)


def create_msfs_ctde_env_concat(difficulty: str = "normal", **kwargs) -> MSFSCTDEWrapper:
    """Create MSFS CTDE environment with concatenated global state"""
    return create_msfs_ctde_env(difficulty=difficulty, global_state_type="concat", **kwargs)


def create_msfs_ctde_env_mean(difficulty: str = "normal", **kwargs) -> MSFSCTDEWrapper:
    """Create MSFS CTDE environment with mean-pooled global state"""
    return create_msfs_ctde_env(difficulty=difficulty, global_state_type="mean", **kwargs)


def create_msfs_ctde_env_max(difficulty: str = "normal", **kwargs) -> MSFSCTDEWrapper:
    """Create MSFS CTDE environment with max-pooled global state"""
    return create_msfs_ctde_env(difficulty=difficulty, global_state_type="max", **kwargs)


def create_msfs_ctde_env_attention(difficulty: str = "normal", **kwargs) -> MSFSCTDEWrapper:
    """Create MSFS CTDE environment with attention-based global state"""
    return create_msfs_ctde_env(difficulty=difficulty, global_state_type="attention", **kwargs)


# Curriculum learning configurations
def create_msfs_ctde_curriculum_stage1(**kwargs) -> MSFSCTDEWrapper:
    """Create MSFS CTDE environment for curriculum stage 1"""
    config = MSFSPresetConfigs.curriculum_stage1()
    return MSFSCTDEWrapper(difficulty="easy", global_state_type="concat", **{**config.to_dict(), **kwargs})


def create_msfs_ctde_curriculum_stage2(**kwargs) -> MSFSCTDEWrapper:
    """Create MSFS CTDE environment for curriculum stage 2"""
    config = MSFSPresetConfigs.curriculum_stage2()
    return MSFSCTDEWrapper(difficulty="normal", global_state_type="concat", **{**config.to_dict(), **kwargs})


def create_msfs_ctde_curriculum_stage3(**kwargs) -> MSFSCTDEWrapper:
    """Create MSFS CTDE environment for curriculum stage 3"""
    config = MSFSPresetConfigs.curriculum_stage3()
    return MSFSCTDEWrapper(difficulty="normal", global_state_type="concat", **{**config.to_dict(), **kwargs})


# Role emergence focused configurations
def create_msfs_ctde_role_emergence(**kwargs) -> MSFSCTDEWrapper:
    """Create MSFS CTDE environment focused on role emergence"""
    config = MSFSPresetConfigs.role_emergence_focus()
    return MSFSCTDEWrapper(difficulty="normal", global_state_type="concat", **{**config.to_dict(), **kwargs})


def create_msfs_ctde_efficiency(**kwargs) -> MSFSCTDEWrapper:
    """Create MSFS CTDE environment focused on efficiency"""
    config = MSFSPresetConfigs.efficiency_focus()
    return MSFSCTDEWrapper(difficulty="hard", global_state_type="concat", **{**config.to_dict(), **kwargs})