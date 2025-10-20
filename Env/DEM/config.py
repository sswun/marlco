"""
DEM Environment Configuration

This module provides configuration classes for the DEM environment.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import numpy as np


@dataclass
class DEMConfig:
    """Configuration class for DEM environment"""

    # Grid and basic settings
    grid_size: int = 12
    max_steps: int = 200
    episode_time_limit: float = 300.0  # Maximum episode time in seconds

    # VIP settings
    vip_initial_pos: tuple = (1, 1)
    vip_target_pos: tuple = (10, 10)
    vip_hp: int = 60
    vip_vision_range: int = 2
    vip_move_cooldown: int = 2

    # Agent settings
    num_agents: int = 3
    agent_hp: int = 50
    agent_damage: int = 10
    agent_range: int = 2
    agent_attack_cooldown: int = 2
    agent_vision_range: int = 4
    agent_initial_positions: List[tuple] = None  # If None, auto-generate

    # Threat settings
    threat_spawn_initial_delay: int = 8  # Steps before first threat spawn
    threat_spawn_base_interval: int = 8  # Base spawn interval
    threat_spawn_healthy_vip_interval: int = 6  # When VIP HP > 40
    threat_spawn_critical_vip_interval: int = 12  # When VIP HP < 20
    max_threats: int = 5
    rusher_probability: float = 0.6  # Probability of spawning rusher vs shooter

    # Rusher settings
    rusher_hp: int = 40
    rusher_damage: int = 8
    rusher_range: int = 1
    rusher_move_range: int = 1
    rusher_attack_cooldown: int = 1

    # Shooter settings
    shooter_hp: int = 30
    shooter_damage: int = 15
    shooter_range: int = 5
    shooter_move_range: int = 0
    shooter_attack_cooldown: int = 3

    # Terrain settings
    river_positions: List[tuple] = None  # If None, auto-generate
    forest_positions: List[tuple] = None  # If None, auto-generate
    forest_damage_reduction: float = 0.7

    # Communication settings
    communication_range: int = 12  # Global for now, can be limited
    max_messages: int = 10
    message_cost: float = 0.01

    # Reward settings
    reward_vip_reach_target: float = 50.0
    reward_vip_death: float = -30.0
    reward_vip_progress: float = 0.2  # Per grid unit closer to target
    reward_threat_killed: float = 3.0
    reward_vip_damage: float = -0.1  # Per HP point
    reward_agent_death: float = -3.0

    # Role emergence rewards
    reward_guard_adjacent: float = 0.05
    reward_guard_missing_penalty: float = -0.02
    reward_body_block: float = 0.5
    reward_vanguard_ahead: float = 0.05
    reward_vanguard_missing_penalty: float = -0.02
    reward_long_range_kill: float = 1.0  # Kill from >=6 units away
    reward_spread_good: float = 0.02  # Average distance [2,5]
    reward_spread_bad: float = -0.01  # Average distance outside [2,5]

    # Movement penalties
    penalty_collision: float = -0.05
    penalty_invalid_action: float = -0.1

    # Visualization settings
    render_mode: str = "rgb_array"  # "human", "rgb_array", or None
    render_fps: int = 4
    render_grid_size: int = 50  # Pixels per grid cell

    # Random seed
    seed: Optional[int] = None

    # Difficulty settings
    difficulty: str = "normal"  # "easy", "normal", "hard"

    def __post_init__(self):
        """Post-initialization setup"""

        # Apply difficulty settings first
        if self.difficulty == "easy":
            self.threat_spawn_base_interval = 12
            self.threat_spawn_healthy_vip_interval = 10
            self.max_threats = 3
            self.rusher_probability = 0.4
            self.vip_hp = 80
            self.agent_hp = 60
            self.grid_size = 10  # Smaller grid
            self.vip_target_pos = (8, 8)  # Closer target

        elif self.difficulty == "hard":
            self.threat_spawn_base_interval = 6
            self.threat_spawn_healthy_vip_interval = 4
            self.max_threats = 8
            self.rusher_probability = 0.8
            self.vip_hp = 40
            self.agent_hp = 40
            self.max_steps = 150  # Shorter time limit
            # Could add moving obstacles in hard mode

        # Generate default positions if not provided (after grid_size is set)
        if self.river_positions is None:
            self.river_positions = self._generate_river_positions()

        if self.forest_positions is None:
            self.forest_positions = self._generate_forest_positions()

        if self.agent_initial_positions is None:
            self.agent_initial_positions = self._generate_agent_positions()

    def _generate_agent_positions(self) -> List[tuple]:
        """Generate default agent positions around VIP"""
        positions = []
        start_positions = [
            (0, 1), (1, 0), (2, 1),
            (1, 2), (0, 0), (2, 0)
        ]

        # Filter valid positions
        for pos in start_positions:
            if (0 <= pos[0] < self.grid_size and
                0 <= pos[1] < self.grid_size and
                pos not in self.river_positions):
                positions.append(pos)

        # Return only num_agents positions
        return positions[:self.num_agents]

    def _generate_river_positions(self) -> List[tuple]:
        """Generate default river positions"""
        positions = []

        # Add some vertical and horizontal rivers
        if self.grid_size >= 10:
            # Vertical river
            river_x = self.grid_size // 2
            for y in range(3, self.grid_size - 2):
                if (river_x, y) not in [self.vip_initial_pos, self.vip_target_pos]:
                    positions.append((river_x, y))

            # Horizontal river
            river_y = self.grid_size // 2
            for x in range(3, self.grid_size - 2):
                if (x, river_y) not in [self.vip_initial_pos, self.vip_target_pos]:
                    positions.append((x, river_y))

        return positions

    def _generate_forest_positions(self) -> List[tuple]:
        """Generate default forest positions"""
        positions = []

        # Forests near start
        start_forests = [(2, 2), (2, 3), (3, 2), (3, 3)]

        # Forests near target
        target_forests = [
            (self.vip_target_pos[0] - 1, self.vip_target_pos[1] - 1),
            (self.vip_target_pos[0] - 1, self.vip_target_pos[1]),
            (self.vip_target_pos[0], self.vip_target_pos[1] - 1),
            (self.vip_target_pos[0], self.vip_target_pos[1])
        ]

        # Additional forests
        additional_forests = [(5, 2), (5, 3), (6, 8), (7, 8)]

        all_forests = start_forests + target_forests + additional_forests

        for forest_pos in all_forests:
            if (0 <= forest_pos[0] < self.grid_size and
                0 <= forest_pos[1] < self.grid_size and
                forest_pos not in self.river_positions and
                forest_pos not in [self.vip_initial_pos, self.vip_target_pos]):
                positions.append(forest_pos)

        return positions

    def validate(self) -> bool:
        """Validate configuration"""
        errors = []

        # Basic grid validation
        if self.grid_size < 8:
            errors.append("Grid size must be at least 8")

        if self.vip_initial_pos[0] >= self.grid_size or self.vip_initial_pos[1] >= self.grid_size:
            errors.append("VIP initial position out of bounds")

        if self.vip_target_pos[0] >= self.grid_size or self.vip_target_pos[1] >= self.grid_size:
            errors.append("VIP target position out of bounds")

        # Agent validation
        if self.num_agents < 1:
            errors.append("Number of agents must be at least 1")

        if len(self.agent_initial_positions) < self.num_agents:
            errors.append("Not enough initial positions for agents")

        # Threat validation
        if self.max_threats < 1:
            errors.append("Max threats must be at least 1")

        if not 0 <= self.rusher_probability <= 1:
            errors.append("Rusher probability must be between 0 and 1")

        # Reward validation
        if self.reward_vip_reach_target <= 0:
            errors.append("VIP reach target reward must be positive")

        if self.reward_vip_death >= 0:
            errors.append("VIP death reward should be negative")

        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(errors))

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'grid_size': self.grid_size,
            'max_steps': self.max_steps,
            'episode_time_limit': self.episode_time_limit,
            'vip_initial_pos': self.vip_initial_pos,
            'vip_target_pos': self.vip_target_pos,
            'vip_hp': self.vip_hp,
            'vip_vision_range': self.vip_vision_range,
            'vip_move_cooldown': self.vip_move_cooldown,
            'num_agents': self.num_agents,
            'agent_hp': self.agent_hp,
            'agent_damage': self.agent_damage,
            'agent_range': self.agent_range,
            'agent_attack_cooldown': self.agent_attack_cooldown,
            'agent_vision_range': self.agent_vision_range,
            'agent_initial_positions': self.agent_initial_positions,
            'threat_spawn_initial_delay': self.threat_spawn_initial_delay,
            'threat_spawn_base_interval': self.threat_spawn_base_interval,
            'threat_spawn_healthy_vip_interval': self.threat_spawn_healthy_vip_interval,
            'threat_spawn_critical_vip_interval': self.threat_spawn_critical_vip_interval,
            'max_threats': self.max_threats,
            'rusher_probability': self.rusher_probability,
            'rusher_hp': self.rusher_hp,
            'rusher_damage': self.rusher_damage,
            'rusher_range': self.rusher_range,
            'rusher_move_range': self.rusher_move_range,
            'rusher_attack_cooldown': self.rusher_attack_cooldown,
            'shooter_hp': self.shooter_hp,
            'shooter_damage': self.shooter_damage,
            'shooter_range': self.shooter_range,
            'shooter_move_range': self.shooter_move_range,
            'shooter_attack_cooldown': self.shooter_attack_cooldown,
            'river_positions': self.river_positions,
            'forest_positions': self.forest_positions,
            'forest_damage_reduction': self.forest_damage_reduction,
            'communication_range': self.communication_range,
            'max_messages': self.max_messages,
            'message_cost': self.message_cost,
            'reward_vip_reach_target': self.reward_vip_reach_target,
            'reward_vip_death': self.reward_vip_death,
            'reward_vip_progress': self.reward_vip_progress,
            'reward_threat_killed': self.reward_threat_killed,
            'reward_vip_damage': self.reward_vip_damage,
            'reward_agent_death': self.reward_agent_death,
            'reward_guard_adjacent': self.reward_guard_adjacent,
            'reward_guard_missing_penalty': self.reward_guard_missing_penalty,
            'reward_body_block': self.reward_body_block,
            'reward_vanguard_ahead': self.reward_vanguard_ahead,
            'reward_vanguard_missing_penalty': self.reward_vanguard_missing_penalty,
            'reward_long_range_kill': self.reward_long_range_kill,
            'reward_spread_good': self.reward_spread_good,
            'reward_spread_bad': self.reward_spread_bad,
            'penalty_collision': self.penalty_collision,
            'penalty_invalid_action': self.penalty_invalid_action,
            'render_mode': self.render_mode,
            'render_fps': self.render_fps,
            'render_grid_size': self.render_grid_size,
            'seed': self.seed,
            'difficulty': self.difficulty
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DEMConfig':
        """Create configuration from dictionary"""
        return cls(**config_dict)

    def create_easy_config(self) -> 'DEMConfig':
        """Create easy difficulty configuration"""
        config = DEMConfig()
        config.difficulty = "easy"
        config.__post_init__()
        return config

    def create_normal_config(self) -> 'DEMConfig':
        """Create normal difficulty configuration"""
        config = DEMConfig()
        config.difficulty = "normal"
        config.__post_init__()
        return config

    def create_hard_config(self) -> 'DEMConfig':
        """Create hard difficulty configuration"""
        config = DEMConfig()
        config.difficulty = "hard"
        config.__post_init__()
        return config