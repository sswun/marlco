"""
CM Environment CTDE Wrapper

This module provides a Centralized Training Decentralized Execution (CTDE) wrapper
for the Collaborative Moving environment, making it compatible with popular MARL
algorithms like QMIX, VDN, and MADDPG.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union

from .env_cm import CooperativeMovingEnv
from .config import CMConfig, get_config_by_name


class CooperativeMovingCTDEEnv:
    """
    CTDE wrapper for the Collaborative Moving environment.

    This wrapper provides the centralized training interface needed by most MARL
    algorithms while maintaining the decentralized execution interface.
    """

    def __init__(self,
                 difficulty: str = "normal",
                 global_state_type: str = "concat",
                 **kwargs):
        """
        Initialize the CTDE environment.

        Args:
            difficulty: Predefined difficulty level
            global_state_type: Type of global state representation
                - "concat": Concatenate all agent observations
                - "mean": Average of all agent observations
                - "max": Element-wise maximum of observations
                - "attention": Attention-based state representation
            **kwargs: Additional configuration overrides
        """
        # Create base environment
        config = get_config_by_name(difficulty)
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        self.config = config
        self.base_env = CooperativeMovingEnv(config=config)
        self.global_state_type = global_state_type

        # Environment properties
        self.n_agents = self.base_env.n_agents
        self.agent_ids = self.base_env.agent_ids
        self.n_actions = self.base_env.n_actions
        self.max_steps = self.base_env.config.max_steps

        # Calculate global state dimension
        if global_state_type == "concat":
            self.global_state_dim = self.base_env.observation_space.shape[0] * self.n_agents
        elif global_state_type in ["mean", "max"]:
            self.global_state_dim = self.base_env.observation_space.shape[0]
        elif global_state_type == "attention":
            # Observation + attention weights
            self.global_state_dim = self.base_env.observation_space.shape[0] + self.n_agents
        else:
            raise ValueError(f"Unknown global_state_type: {global_state_type}")

        # Current state
        self.observations: Optional[Dict[str, np.ndarray]] = None
        self.global_state: Optional[np.ndarray] = None

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Reset the environment and return initial observations and global state.

        Returns:
            observations: Dictionary mapping agent IDs to observations
            global_state: Centralized state representation for training
        """
        observations = self.base_env.reset(seed=seed)
        self.observations = observations
        self.global_state = self._compute_global_state(observations)

        return observations, self.global_state

    def step(self, actions: Dict[str, int]) -> Tuple[
        Dict[str, np.ndarray],  # observations
        Dict[str, float],       # rewards
        Dict[str, bool],        # dones
        Dict[str, Any]          # info
    ]:
        """
        Execute one step in the environment.

        Args:
            actions: Dictionary mapping agent IDs to actions

        Returns:
            observations: New observations for all agents
            rewards: Rewards for all agents
            dones: Done flags for each agent
            info: Additional environment information (includes global_state)
        """
        observations, rewards, dones, info = self.base_env.step(actions)
        self.observations = observations
        self.global_state = self._compute_global_state(observations)

        # Add global state to info
        info['global_state'] = self.global_state

        return observations, rewards, dones, info

    def _compute_global_state(self, observations: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute centralized global state from agent observations.

        Args:
            observations: Dictionary of agent observations

        Returns:
            Global state representation
        """
        obs_list = [observations[agent_id] for agent_id in self.agent_ids]
        obs_array = np.stack(obs_list)  # Shape: (n_agents, obs_dim)

        if self.global_state_type == "concat":
            # Concatenate all observations
            return obs_array.flatten()

        elif self.global_state_type == "mean":
            # Element-wise mean
            return np.mean(obs_array, axis=0)

        elif self.global_state_type == "max":
            # Element-wise maximum
            return np.max(obs_array, axis=0)

        elif self.global_state_type == "attention":
            # Simple attention-like representation
            mean_obs = np.mean(obs_array, axis=0)
            max_obs = np.max(obs_array, axis=0)
            min_obs = np.min(obs_array, axis=0)

            # Add statistical features
            attention_features = np.concatenate([
                mean_obs,
                max_obs - min_obs,  # Range
                np.std(obs_array, axis=0)  # Standard deviation
            ])

            return attention_features

        else:
            raise ValueError(f"Unknown global_state_type: {self.global_state_type}")

    def get_global_state(self) -> np.ndarray:
        """Get the current global state."""
        if self.global_state is None:
            raise RuntimeError("Environment must be reset before getting global state")
        return self.global_state

    def get_avail_actions(self, agent_id: str) -> List[int]:
        """Get available actions for a specific agent."""
        return self.base_env.get_avail_actions(agent_id)

    def get_env_info(self) -> Dict[str, Any]:
        """Get environment information for training algorithms."""
        base_info = self.base_env.get_env_info()
        base_info.update({
            'global_state_dim': self.global_state_dim,
            'global_state_type': self.global_state_type,
            'episode_limit': self.max_steps
        })
        return base_info

    def render(self) -> Optional[np.ndarray]:
        """Render the environment."""
        return self.base_env.render()

    def close(self):
        """Close the environment."""
        self.base_env.close()

    def seed(self, seed: int):
        """Set random seed."""
        self.base_env.reset(seed=seed)

    def __getattr__(self, name):
        """Delegate attribute access to base environment."""
        return getattr(self.base_env, name)


class CMCTDEWrapper:
    """
    Alternative CTDE wrapper implementation for compatibility with different frameworks.
    """

    def __init__(self, base_env: CooperativeMovingEnv, global_state_type: str = "concat"):
        """
        Initialize the CTDE wrapper.

        Args:
            base_env: Base CM environment
            global_state_type: Type of global state representation
        """
        self.base_env = base_env
        self.global_state_type = global_state_type

        # Environment properties
        self.n_agents = base_env.n_agents
        self.agent_ids = base_env.agent_ids
        self.n_actions = base_env.n_actions
        self.max_steps = base_env.config.max_steps

        # Calculate global state dimension
        if global_state_type == "concat":
            self.global_state_dim = base_env.observation_space.shape[0] * self.n_agents
        elif global_state_type in ["mean", "max"]:
            self.global_state_dim = base_env.observation_space.shape[0]
        elif global_state_type == "attention":
            self.global_state_dim = base_env.observation_space.shape[0] + self.n_agents * 3
        else:
            raise ValueError(f"Unknown global_state_type: {global_state_type}")

        # Current state
        self.observations: Optional[Dict[str, np.ndarray]] = None
        self.global_state: Optional[np.ndarray] = None

    def reset(self, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Reset environment and return observations only."""
        observations = self.base_env.reset(seed=seed)
        self.observations = observations
        self.global_state = self._compute_global_state(observations)
        return observations

    def step(self, actions: Dict[str, int]) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, Any]
    ]:
        """Execute one step in the environment."""
        observations, rewards, dones, info = self.base_env.step(actions)
        self.observations = observations
        self.global_state = self._compute_global_state(observations)

        info['global_state'] = self.global_state

        return observations, rewards, dones, info

    def _compute_global_state(self, observations: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute global state from observations."""
        obs_list = [observations[agent_id] for agent_id in self.agent_ids]
        obs_array = np.stack(obs_list)

        if self.global_state_type == "concat":
            return obs_array.flatten()
        elif self.global_state_type == "mean":
            return np.mean(obs_array, axis=0)
        elif self.global_state_type == "max":
            return np.max(obs_array, axis=0)
        elif self.global_state_type == "attention":
            mean_obs = np.mean(obs_array, axis=0)
            max_obs = np.max(obs_array, axis=0)
            min_obs = np.min(obs_array, axis=0)
            std_obs = np.std(obs_array, axis=0)
            return np.concatenate([mean_obs, max_obs, min_obs, std_obs])
        else:
            raise ValueError(f"Unknown global_state_type: {self.global_state_type}")

    def get_global_state(self) -> np.ndarray:
        """Get current global state."""
        if self.global_state is None:
            raise RuntimeError("Environment must be reset first")
        return self.global_state

    def get_avail_actions(self, agent_id: str) -> List[int]:
        """Get available actions for agent."""
        return self.base_env.get_avail_actions(agent_id)

    def get_env_info(self) -> Dict[str, Any]:
        """Get environment info."""
        info = self.base_env.get_env_info()
        info.update({
            'global_state_dim': self.global_state_dim,
            'global_state_type': self.global_state_type
        })
        return info

    def render(self) -> Optional[np.ndarray]:
        """Render environment."""
        return self.base_env.render()

    def close(self):
        """Close environment."""
        self.base_env.close()


# Factory functions for easy CTDE environment creation
def create_cm_ctde_env(difficulty: str = "normal_ctde",
                      global_state_type: str = "concat",
                      **kwargs) -> CooperativeMovingCTDEEnv:
    """
    Create a CTDE CM environment.

    Args:
        difficulty: Predefined difficulty level (use _ctde suffix for CTDE-optimized configs)
        global_state_type: Type of global state representation
        **kwargs: Additional configuration overrides

    Returns:
        CTDE environment instance
    """
    return CooperativeMovingCTDEEnv(
        difficulty=difficulty,
        global_state_type=global_state_type,
        **kwargs
    )


def wrap_cm_env_to_ctde(base_env: CooperativeMovingEnv,
                       global_state_type: str = "concat") -> CMCTDEWrapper:
    """
    Wrap an existing CM environment to CTDE format.

    Args:
        base_env: Existing CM environment
        global_state_type: Type of global state representation

    Returns:
        CTDE wrapper instance
    """
    return CMCTDEWrapper(base_env, global_state_type)


# Compatibility functions for different MARL frameworks
class CMEnvironmentAdapter:
    """
    Adapter class to make CM environment compatible with different MARL frameworks.
    """

    def __init__(self, env):
        self.env = env
        self.n_agents = env.n_agents
        self.agent_ids = env.agent_ids
        self.n_actions = env.n_actions
        self.max_steps = env.max_steps

    def step(self, actions):
        """Step function compatible with most MARL frameworks."""
        if isinstance(actions, np.ndarray):
            # Convert numpy array to dict
            action_dict = {}
            for i, agent_id in enumerate(self.agent_ids):
                if i < len(actions):
                    action_dict[agent_id] = int(actions[i])
                else:
                    action_dict[agent_id] = 0  # Default action
            actions = action_dict

        obs, rewards, done, info = self.env.step(actions)
        return obs, rewards, done, info

    def reset(self):
        """Reset function compatible with most MARL frameworks."""
        obs, global_state = self.env.reset()
        return obs, global_state

    def get_obs(self):
        """Get current observations."""
        if hasattr(self.env, 'observations') and self.env.observations:
            return self.env.observations
        else:
            raise RuntimeError("No observations available. Call reset() first.")

    def get_state(self):
        """Get current global state."""
        if hasattr(self.env, 'global_state') and self.env.global_state is not None:
            return self.env.global_state
        else:
            raise RuntimeError("No global state available. Call reset() first.")

    def get_avail_agent_actions(self, agent_id):
        """Get available actions for specific agent."""
        return self.env.get_avail_actions(agent_id)

    def get_obs_size(self):
        """Get observation size."""
        return self.env.base_env.observation_space.shape[0]

    def get_state_size(self):
        """Get global state size."""
        return self.env.global_state_dim

    def get_total_actions(self):
        """Get total number of actions."""
        return self.n_actions

    def get_env_info(self):
        """Get environment information."""
        return self.env.get_env_info()

    def render(self):
        """Render environment."""
        return self.env.render()

    def close(self):
        """Close environment."""
        self.env.close()


def create_cm_adapter(difficulty: str = "normal_ctde",
                     global_state_type: str = "concat",
                     **kwargs) -> CMEnvironmentAdapter:
    """
    Create a CM environment adapter for MARL frameworks.

    Args:
        difficulty: Difficulty level
        global_state_type: Global state representation type
        **kwargs: Additional configuration

    Returns:
        Environment adapter instance
    """
    ctde_env = create_cm_ctde_env(difficulty, global_state_type, **kwargs)
    return CMEnvironmentAdapter(ctde_env)


if __name__ == "__main__":
    # Test CTDE environment
    print("Testing CTDE Environment...")

    # Test different global state types
    for state_type in ["concat", "mean", "max", "attention"]:
        print(f"\nTesting global_state_type: {state_type}")

        env = create_cm_ctde_env(
            difficulty="easy_ctde",
            global_state_type=state_type
        )

        print(f"Environment info: {env.get_env_info()}")

        # Reset environment
        obs, global_state = env.reset()
        print(f"Global state shape: {global_state.shape}")

        # Take a few random steps
        for step in range(3):
            actions = {}
            for agent_id in env.agent_ids:
                avail_actions = env.get_avail_actions(agent_id)
                actions[agent_id] = np.random.choice(avail_actions)

            obs, rewards, done, info = env.step(actions)
            print(f"Step {step}: global_state shape={info['global_state'].shape}")

        env.close()

    print("\nCTDE Environment test completed successfully!")