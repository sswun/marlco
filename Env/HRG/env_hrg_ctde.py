"""
HRG Environment CTDE Wrapper

This module provides a CTDE (Centralized Training, Decentralized Execution)
wrapper for the HRG environment, making it compatible with multi-agent
reinforcement learning algorithms like QMIX, VDN, etc.
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
import logging

from .env_hrg import HRGEnv, HRGConfig, create_hrg_env

# Configure logging
logger = logging.getLogger(__name__)


class HRGCTDEWrapper(HRGEnv):
    """
    CTDE-compatible wrapper for HRG environment

    This wrapper extends the base HRG environment to provide CTDE-specific
    functionality including global state representation and standardized
    interfaces for centralized training algorithms.
    """

    def __init__(self,
                 difficulty: str = "normal",
                 global_state_type: str = "concat",
                 global_state_dim: Optional[int] = None,
                 **kwargs):
        """
        Initialize CTDE wrapper for HRG environment

        Args:
            difficulty: Difficulty level ("easy", "normal", "hard")
            global_state_type: Type of global state representation
                ("concat", "mean", "max", "attention")
            global_state_dim: Dimension of global state (if None, auto-calculated)
            **kwargs: Additional environment configuration
        """
        # Initialize base environment
        config = HRGConfig(**kwargs)
        super().__init__(config)

        # CTDE-specific settings
        self.global_state_type = global_state_type
        self._global_state_dim = global_state_dim

        # Setup global state dimension
        self._setup_global_state_dim()

        logger.info(f"HRG-CTDE environment initialized - Difficulty: {difficulty}, "
                   f"Global state type: {global_state_type}, Dimension: {self._global_state_dim}")

    def _setup_global_state_dim(self):
        """Setup global state dimension based on type"""
        if self._global_state_dim is not None:
            return

        if self.global_state_type == "concat":
            # Concatenate all agent observations (optimized to 60 dims)
            single_obs_dim = 60
            self._global_state_dim = single_obs_dim * self.n_agents
        elif self.global_state_type in ["mean", "max"]:
            # Mean or max pooling over observations (optimized to 60 dims)
            self._global_state_dim = 60
        elif self.global_state_type == "attention":
            # Attention mechanism with agent count info (optimized to 60 dims)
            self._global_state_dim = 60 + self.n_agents
        else:
            # Default to concatenation (optimized)
            self._global_state_dim = 60 * self.n_agents

    @property
    def global_state_dim(self) -> int:
        """Get global state dimension"""
        return self._global_state_dim

    def get_global_state(self) -> np.ndarray:
        """
        Get global state representation for centralized training

        Returns:
            np.ndarray: Global state representation
        """
        try:
            # Get all agent observations
            observations = {}
            for agent_id in self.agent_ids:
                observations[agent_id] = self._get_observation(agent_id)

            # Generate global state based on type
            if self.global_state_type == "concat":
                return self._concat_global_state(observations)
            elif self.global_state_type == "mean":
                return self._mean_global_state(observations)
            elif self.global_state_type == "max":
                return self._max_global_state(observations)
            elif self.global_state_type == "attention":
                return self._attention_global_state(observations)
            else:
                return self._concat_global_state(observations)

        except Exception as e:
            logger.warning(f"Failed to generate global state: {e}")
            return np.zeros(self.global_state_dim)

    def _concat_global_state(self, observations: Dict[str, np.ndarray]) -> np.ndarray:
        """Concatenate all observations to form global state"""
        state_parts = []
        for agent_id in sorted(self.agent_ids):
            if agent_id in observations:
                obs = observations[agent_id]
                state_parts.append(obs.flatten())
            else:
                # Use zeros if observation not available (optimized to 60 dims)
                state_parts.append(np.zeros(60))

        global_state = np.concatenate(state_parts)

        # Ensure correct dimension
        if len(global_state) != self.global_state_dim:
            if len(global_state) < self.global_state_dim:
                padding = np.zeros(self.global_state_dim - len(global_state))
                global_state = np.concatenate([global_state, padding])
            else:
                global_state = global_state[:self.global_state_dim]

        return global_state

    def _mean_global_state(self, observations: Dict[str, np.ndarray]) -> np.ndarray:
        """Generate global state by averaging observations"""
        obs_list = []
        for agent_id in sorted(self.agent_ids):
            if agent_id in observations:
                obs = observations[agent_id]
                obs_list.append(obs.flatten())

        if obs_list:
            obs_array = np.array(obs_list)
            mean_obs = np.mean(obs_array, axis=0)

            # Ensure correct dimension
            if len(mean_obs) != self.global_state_dim:
                if len(mean_obs) < self.global_state_dim:
                    padding = np.zeros(self.global_state_dim - len(mean_obs))
                    mean_obs = np.concatenate([mean_obs, padding])
                else:
                    mean_obs = mean_obs[:self.global_state_dim]

            return mean_obs

        return np.zeros(self.global_state_dim)

    def _max_global_state(self, observations: Dict[str, np.ndarray]) -> np.ndarray:
        """Generate global state by taking maximum over observations"""
        obs_list = []
        for agent_id in sorted(self.agent_ids):
            if agent_id in observations:
                obs = observations[agent_id]
                obs_list.append(obs.flatten())

        if obs_list:
            obs_array = np.array(obs_list)
            max_obs = np.max(obs_array, axis=0)

            # Ensure correct dimension
            if len(max_obs) != self.global_state_dim:
                if len(max_obs) < self.global_state_dim:
                    padding = np.zeros(self.global_state_dim - len(max_obs))
                    max_obs = np.concatenate([max_obs, padding])
                else:
                    max_obs = max_obs[:self.global_state_dim]

            return max_obs

        return np.zeros(self.global_state_dim)

    def _attention_global_state(self, observations: Dict[str, np.ndarray]) -> np.ndarray:
        """Generate global state using attention mechanism"""
        obs_list = []
        for agent_id in sorted(self.agent_ids):
            if agent_id in observations:
                obs = observations[agent_id]
                obs_list.append(obs.flatten())

        if obs_list:
            # Calculate attention weights based on L2 norm
            obs_array = np.array(obs_list)
            attention_weights = np.linalg.norm(obs_array, axis=1)
            attention_weights = np.exp(attention_weights) / np.sum(np.exp(attention_weights))

            # Weighted average
            weighted_obs = np.zeros_like(obs_array[0])
            for i, weight in enumerate(attention_weights):
                weighted_obs += weight * obs_array[i]

            # Add agent count information
            agent_count_info = np.array([self.n_agents / 10.0])  # Normalized
            global_state = np.concatenate([weighted_obs, agent_count_info])

            # Ensure correct dimension
            if len(global_state) != self.global_state_dim:
                if len(global_state) < self.global_state_dim:
                    padding = np.zeros(self.global_state_dim - len(global_state))
                    global_state = np.concatenate([global_state, padding])
                else:
                    global_state = global_state[:self.global_state_dim]

            return global_state

        return np.zeros(self.global_state_dim)

    def reset(self) -> Dict[str, np.ndarray]:
        """
        Reset environment and return initial observations

        Returns:
            Dict[str, np.ndarray]: Initial observations for each agent
        """
        observations = super().reset()

        # Store observations for global state generation
        self._last_observations = observations.copy()

        return observations

    def step(self, actions: Dict[str, Union[int, np.ndarray]]) -> Tuple[
        Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, Any]
    ]:
        """
        Execute one step of the environment

        Args:
            actions: Dictionary of actions for each agent

        Returns:
            Tuple of (observations, rewards, dones, infos)
        """
        observations, rewards, dones, infos = super().step(actions)

        # Store observations for global state generation
        self._last_observations = observations.copy()

        # Add global state to info
        if 'global_state' not in infos:
            infos['global_state'] = self.get_global_state()

        return observations, rewards, dones, infos

    def get_env_info(self) -> Dict[str, Any]:
        """
        Get environment information for algorithm configuration

        Returns:
            Dict[str, Any]: Environment information
        """
        return {
            'env_name': 'HRG-CTDE',
            'n_agents': self.n_agents,
            'agent_ids': self.agent_ids,
            'obs_dims': self.obs_dims,
            'act_dims': self.act_dims,
            'global_state_dim': self.global_state_dim,
            'episode_limit': self.config.max_steps,
            'global_state_type': self.global_state_type
        }


class HRGCTDEConfig:
    """Configuration presets for HRG-CTDE environment"""

    @staticmethod
    def easy_ctde(**kwargs):
        """Easy difficulty CTDE configuration"""
        return {
            'difficulty': 'easy',
            'grid_size': 8,
            'max_steps': 300,
            'num_obstacles': 0,
            'num_gold': 2,
            'num_wood': 15,
            **kwargs
        }

    @staticmethod
    def normal_ctde(**kwargs):
        """Normal difficulty CTDE configuration"""
        return {
            'difficulty': 'normal',
            'grid_size': 10,
            'max_steps': 200,
            'num_obstacles': 10,
            'num_gold': 3,
            'num_wood': 10,
            **kwargs
        }

    @staticmethod
    def hard_ctde(**kwargs):
        """Hard difficulty CTDE configuration"""
        return {
            'difficulty': 'hard',
            'grid_size': 12,
            'max_steps': 150,
            'num_obstacles': 20,
            'num_gold': 4,
            'num_wood': 8,
            **kwargs
        }


def create_hrg_ctde_env(config_name: str = "normal_ctde",
                        global_state_type: str = "concat",
                        **kwargs) -> HRGCTDEWrapper:
    """
    Convenience function to create HRG-CTDE environment with predefined configurations

    Args:
        config_name: Name of configuration preset
        global_state_type: Type of global state representation
        **kwargs: Additional configuration parameters

    Returns:
        HRGCTDEWrapper: Configured CTDE environment
    """
    config_map = {
        "easy_ctde": HRGCTDEConfig.easy_ctde,
        "normal_ctde": HRGCTDEConfig.normal_ctde,
        "hard_ctde": HRGCTDEConfig.hard_ctde,
    }

    if config_name not in config_map:
        raise ValueError(f"Unknown configuration: {config_name}. "
                        f"Available: {list(config_map.keys())}")

    config = config_map[config_name](**kwargs)
    return HRGCTDEWrapper(
        global_state_type=global_state_type,
        **config
    )


def test_hrg_ctde_wrapper():
    """Test the HRG-CTDE wrapper functionality"""
    print("Testing HRG-CTDE wrapper...")

    try:
        # Test different global state types
        for global_state_type in ["concat", "mean", "max", "attention"]:
            print(f"\nTesting global state type: {global_state_type}")

            env = create_hrg_ctde_env(
                "normal_ctde",
                global_state_type=global_state_type
            )

            print(f"Environment created successfully")
            print(f"Number of agents: {env.n_agents}")
            print(f"Global state dimension: {env.global_state_dim}")

            # Reset environment
            observations = env.reset()
            print(f"Reset successful, observation shapes: {[obs.shape for obs in observations.values()]}")

            # Get global state
            global_state = env.get_global_state()
            print(f"Global state shape: {global_state.shape}")
            print(f"Global state non-zero elements: {np.count_nonzero(global_state)}")

            # Execute a few steps
            for step in range(3):
                actions = {}
                for agent_id in env.agent_ids:
                    avail_actions = env.get_avail_actions(agent_id)
                    if avail_actions:
                        action = np.random.choice(avail_actions)
                    else:
                        action = np.random.randint(0, 8)
                    actions[agent_id] = action

                observations, rewards, dones, info = env.step(actions)

                if 'global_state' in info:
                    print(f"Step {step+1}: Global state in info, shape: {info['global_state'].shape}")

                print(f"Step {step+1}: Rewards: {rewards}")

                if any(dones.values()):
                    print("Episode ended early")
                    break

            env.close()
            print(f"{global_state_type} type test passed")

        print("\nAll HRG-CTDE wrapper tests passed!")

    except Exception as e:
        print(f"HRG-CTDE wrapper test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run tests
    test_hrg_ctde_wrapper()