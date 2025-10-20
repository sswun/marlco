"""
CM Environment Configuration

This module defines configuration classes for the Collaborative Moving environment,
including different difficulty levels and experimental settings.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import numpy as np


@dataclass
class CMConfig:
    """Base configuration for the CM environment."""

    # Grid and world settings
    grid_size: int = 7
    box_size: int = 2
    goal_size: int = 2

    # Agent settings
    n_agents: int = 2
    agent_ids: Optional[List[str]] = None

    # Episode settings
    max_steps: int = 100
    time_penalty: float = -0.01

    # Box pushing mechanics
    push_success_probs: Dict[int, float] = None  # {n_cooperating: success_prob}
    cooperation_reward: float = 0.02
    box_move_reward_scale: float = 0.5

    # Goal rewards
    goal_reached_reward: float = 50.0  # 大幅增加目标奖励
    distance_reward_scale: float = 0.5

    # Collision settings
    agent_collision_penalty: float = -0.1
    box_collision_penalty: float = -0.05

    # Observation settings
    normalize_observations: bool = True
    include_relative_positions: bool = True

    # Rendering
    render_mode: Optional[str] = None  # None, 'human', 'rgb_array'
    render_fps: int = 4

    # Random seed
    seed: Optional[int] = None

    def __post_init__(self):
        """Initialize derived fields and validate configuration."""
        # Set default push success probabilities if not provided
        if self.push_success_probs is None:
            self.push_success_probs = {
                1: 0.5,  # 50% chance with 1 agent
                2: 0.75, # 75% chance with 2 agents
                3: 0.9,  # 90% chance with 3 agents
                4: 1.0   # 100% chance with 4 agents
            }

        # Generate agent IDs if not provided
        if self.agent_ids is None:
            self.agent_ids = [f"agent_{i}" for i in range(self.n_agents)]
        elif len(self.agent_ids) != self.n_agents:
            raise ValueError(f"Number of agent IDs ({len(self.agent_ids)}) must match n_agents ({self.n_agents})")

        # Validate configuration
        self._validate_config()

    def _validate_config(self):
        """Validate configuration parameters."""
        if self.grid_size < 5:
            raise ValueError("Grid size must be at least 5x5")

        if self.grid_size < self.box_size + 2:
            raise ValueError("Grid must be large enough to accommodate box and movement")

        if self.n_agents < 1 or self.n_agents > 4:
            raise ValueError("Number of agents must be between 1 and 4")

        if self.max_steps <= 0:
            raise ValueError("Max steps must be positive")

        if not (0 <= self.push_success_probs.get(1, 0) <= 1):
            raise ValueError("Push success probabilities must be between 0 and 1")

    def copy(self) -> 'CMConfig':
        """Create a copy of this configuration."""
        return CMConfig(
            grid_size=self.grid_size,
            box_size=self.box_size,
            goal_size=self.goal_size,
            n_agents=self.n_agents,
            agent_ids=self.agent_ids.copy(),
            max_steps=self.max_steps,
            time_penalty=self.time_penalty,
            push_success_probs=self.push_success_probs.copy(),
            cooperation_reward=self.cooperation_reward,
            box_move_reward_scale=self.box_move_reward_scale,
            goal_reached_reward=self.goal_reached_reward,
            distance_reward_scale=self.distance_reward_scale,
            agent_collision_penalty=self.agent_collision_penalty,
            box_collision_penalty=self.box_collision_penalty,
            normalize_observations=self.normalize_observations,
            include_relative_positions=self.include_relative_positions,
            render_mode=self.render_mode,
            render_fps=self.render_fps,
            seed=self.seed
        )


# Predefined difficulty configurations
def get_easy_config() -> CMConfig:
    """Get easy difficulty configuration."""
    return CMConfig(
        grid_size=7,
        n_agents=2,
        max_steps=100,
        push_success_probs={1: 0.7, 2: 0.9, 3: 1.0, 4: 1.0},
        cooperation_reward=0.5,  # 增加合作奖励
        time_penalty=-0.1,  # 增加时间惩罚
        goal_reached_reward=80.0,  # 大幅增加目标奖励
        distance_reward_scale=2.0,  # 增加距离奖励
        box_move_reward_scale=5.0  # 增加移动奖励
    )


def get_normal_config() -> CMConfig:
    """Get normal difficulty configuration."""
    return CMConfig(
        grid_size=7,
        n_agents=2,
        max_steps=100,
        push_success_probs={1: 0.5, 2: 0.75, 3: 0.9, 4: 1.0},
        cooperation_reward=1.5,  # 稍微增加合作奖励
        time_penalty=-0.3,  # 增加时间惩罚以降低随机奖励
        goal_reached_reward=50.0,  # 保持主要奖励信号
        distance_reward_scale=0.3,  # 降低距离奖励
        box_move_reward_scale=1.0  # 降低移动奖励
    )


def get_hard_config() -> CMConfig:
    """Get hard difficulty configuration."""
    return CMConfig(
        grid_size=9,
        n_agents=3,
        max_steps=150,
        push_success_probs={1: 0.3, 2: 0.6, 3: 0.85, 4: 1.0},
        cooperation_reward=0.2,  # 增加合作奖励
        time_penalty=-0.2,  # 增加时间惩罚
        goal_reached_reward=100.0,  # 大幅增加目标奖励
        distance_reward_scale=1.0,  # 增加距离奖励
        box_move_reward_scale=2.0  # 增加移动奖励
    )


def get_debug_config() -> CMConfig:
    """Get debug configuration for quick testing."""
    return CMConfig(
        grid_size=5,
        n_agents=2,
        max_steps=50,
        push_success_probs={1: 0.8, 2: 1.0, 3: 1.0, 4: 1.0},
        cooperation_reward=0.05,
        time_penalty=-0.005,
        goal_reached_reward=20.0,
        distance_reward_scale=1.0,
        seed=42
    )


# Specialized configurations for different experiments
def get_cooperation_test_config() -> CMConfig:
    """Configuration designed to test cooperation mechanisms."""
    return CMConfig(
        grid_size=7,
        n_agents=3,
        max_steps=100,
        push_success_probs={1: 0.2, 2: 0.7, 3: 0.95, 4: 1.0},  # Strong cooperation incentive
        cooperation_reward=0.05,  # Higher cooperation reward
        time_penalty=-0.01,
        goal_reached_reward=12.0,
        distance_reward_scale=0.4
    )


def get_single_agent_config() -> CMConfig:
    """Configuration for single agent testing."""
    return CMConfig(
        grid_size=5,
        n_agents=1,
        max_steps=80,
        push_success_probs={1: 0.8, 2: 1.0, 3: 1.0, 4: 1.0},
        cooperation_reward=0.0,  # No cooperation reward for single agent
        time_penalty=-0.01,
        goal_reached_reward=15.0,
        distance_reward_scale=0.7
    )


def get_multi_agent_config() -> CMConfig:
    """Configuration for multiple agents (4 agents)."""
    return CMConfig(
        grid_size=9,
        n_agents=4,
        max_steps=120,
        push_success_probs={1: 0.3, 2: 0.6, 3: 0.85, 4: 1.0},
        cooperation_reward=0.025,
        time_penalty=-0.012,
        goal_reached_reward=12.0,
        distance_reward_scale=0.4,
        agent_collision_penalty=-0.15  # Higher collision penalty for more agents
    )


# Configuration registry for easy access
CONFIG_REGISTRY = {
    "easy": get_easy_config,
    "normal": get_normal_config,
    "hard": get_hard_config,
    "debug": get_debug_config,
    "cooperation_test": get_cooperation_test_config,
    "single_agent": get_single_agent_config,
    "multi_agent": get_multi_agent_config
}


def get_config_by_name(name: str) -> CMConfig:
    """Get configuration by name from registry."""
    if name not in CONFIG_REGISTRY:
        available = list(CONFIG_REGISTRY.keys())
        raise ValueError(f"Unknown config '{name}'. Available configs: {available}")

    return CONFIG_REGISTRY[name]()


def list_available_configs() -> List[str]:
    """List all available configuration names."""
    return list(CONFIG_REGISTRY.keys())


# CTDE-specific configurations
def get_easy_ctde_config() -> CMConfig:
    """Easy configuration optimized for CTDE algorithms."""
    config = get_easy_config()
    config.normalize_observations = True
    config.include_relative_positions = True
    return config


def get_normal_ctde_config() -> CMConfig:
    """Normal configuration optimized for CTDE algorithms."""
    config = get_normal_config()
    config.normalize_observations = True
    config.include_relative_positions = True
    return config


def get_hard_ctde_config() -> CMConfig:
    """Hard configuration optimized for CTDE algorithms."""
    config = get_hard_config()
    config.normalize_observations = True
    config.include_relative_positions = True
    return config


# Add CTDE configs to registry
CONFIG_REGISTRY.update({
    "easy_ctde": get_easy_ctde_config,
    "normal_ctde": get_normal_ctde_config,
    "hard_ctde": get_hard_ctde_config
})


class ConfigManager:
    """Utility class for managing environment configurations."""

    @staticmethod
    def create_config(difficulty: str = "normal", **kwargs) -> CMConfig:
        """Create a configuration with optional parameter overrides."""
        config = get_config_by_name(difficulty)

        # Apply any overrides
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                raise ValueError(f"Unknown config parameter: {key}")

        # Re-validate after modifications
        config._validate_config()

        return config

    @staticmethod
    def get_all_configs() -> Dict[str, CMConfig]:
        """Get all available configurations."""
        return {name: get_config_by_name(name) for name in CONFIG_REGISTRY.keys()}

    @staticmethod
    def compare_configs(config_names: List[str]) -> Dict[str, Dict[str, Any]]:
        """Compare multiple configurations side by side."""
        configs = {name: get_config_by_name(name) for name in config_names}

        comparison = {}
        fields = [
            'grid_size', 'n_agents', 'max_steps', 'push_success_probs',
            'cooperation_reward', 'time_penalty', 'goal_reached_reward',
            'distance_reward_scale', 'agent_collision_penalty'
        ]

        for field in fields:
            comparison[field] = {
                name: getattr(config, field) for name, config in configs.items()
            }

        return comparison


if __name__ == "__main__":
    # Example usage and testing
    print("Available configurations:")
    for name in list_available_configs():
        print(f"  - {name}")

    print("\nEasy config example:")
    easy_config = get_easy_config()
    print(f"  Grid size: {easy_config.grid_size}")
    print(f"  Agents: {easy_config.n_agents}")
    print(f"  Max steps: {easy_config.max_steps}")

    print("\nConfig comparison:")
    comparison = ConfigManager.compare_configs(["easy", "normal", "hard"])
    for field, values in comparison.items():
        print(f"  {field}: {values}")