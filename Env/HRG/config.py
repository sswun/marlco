"""
HRG Environment Configuration Module

This module provides configuration classes and preset configurations
for different experimental setups in the HRG environment.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from .core import AgentType


@dataclass
class HRGEnvironmentConfig:
    """Comprehensive configuration for HRG environment"""

    # Grid and world settings
    grid_size: int = 10
    max_steps: int = 200
    num_obstacles: int = 10
    seed: Optional[int] = None

    # Resource settings
    num_gold: int = 3
    num_wood: int = 10
    gold_respawn_time: int = 50
    wood_respawn_time: int = 50
    gold_cluster_radius: int = 2
    gold_cluster_center: tuple = (8, 8)
    wood_min_distance: int = 2

    # Agent configuration
    agent_config: Dict[str, List[AgentType]] = field(default_factory=lambda: {
        'scouts': [AgentType.SCOUT, AgentType.SCOUT],
        'workers': [AgentType.WORKER, AgentType.WORKER, AgentType.WORKER],
        'transporters': [AgentType.TRANSPORTER]
    })

    # Agent abilities (can be overridden for experiments)
    scout_vision_range: int = 5
    scout_move_speed: float = 2.0
    scout_energy_consumption: float = 0.05

    worker_vision_range: int = 3
    worker_move_speed: float = 1.0
    worker_carry_capacity: int = 2
    worker_gather_time: int = 2
    worker_energy_consumption_move: float = 0.02
    worker_energy_consumption_gather: float = 0.08

    transporter_vision_range: int = 4
    transporter_move_speed: float = 1.5
    transporter_carry_capacity: int = 5
    transporter_energy_consumption_move: float = 0.03
    transporter_energy_consumption_transfer: float = 0.1

    # Reward settings
    gold_value: float = 10.0
    wood_value: float = 2.0
    deposit_reward_ratio: float = 0.5  # 50% of full value for depositing
    transfer_reward_ratio: float = 0.05  # 5% of value for transferring
    gather_reward_ratio: float = 0.1  # 10% of value for gathering
    step_penalty: float = 0.01
    invalid_move_penalty: float = 0.1
    resource_diversity_bonus: float = 0.1
    wood_diversity_bonus: float = 0.05

    # Visualization settings
    render_mode: str = "rgb_array"  # "human", "rgb_array", or None
    render_fps: int = 4
    show_vision_ranges: bool = True
    show_agent_ids: bool = True
    show_inventory: bool = True

    # Communication settings (for future use)
    enable_communication: bool = False
    message_dim: int = 16
    max_message_history: int = 3
    communication_cost: float = 0.01

    # Learning settings
    curriculum_difficulty: str = "normal"  # "easy", "normal", "hard"
    curriculum_stage: int = 1
    adaptive_difficulty: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, AgentType):
                result[key] = value.value
            elif isinstance(value, list) and value and isinstance(value[0], AgentType):
                result[key] = [agent_type.value for agent_type in value]
            else:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'HRGEnvironmentConfig':
        """Create config from dictionary"""
        # Convert agent types back to enums
        agent_config = config_dict.get('agent_config', {})
        converted_agent_config = {}
        for role, agent_list in agent_config.items():
            if isinstance(agent_list, list) and agent_list:
                converted_agent_config[role] = [AgentType(agent_id) for agent_id in agent_list]
            else:
                converted_agent_config[role] = agent_list

        config_dict['agent_config'] = converted_agent_config
        return cls(**config_dict)


class HRGPresetConfigs:
    """Preset configurations for different experimental scenarios"""

    @staticmethod
    def easy() -> HRGEnvironmentConfig:
        """Easy configuration for initial training"""
        return HRGEnvironmentConfig(
            grid_size=8,
            max_steps=300,
            num_obstacles=0,
            num_gold=2,
            num_wood=15,
            gold_cluster_center=(6, 6),
            worker_vision_range=4,  # Increased vision
            worker_carry_capacity=3,  # Increased capacity
            gather_reward_ratio=0.2,  # Higher immediate rewards
            step_penalty=0.005,  # Lower time penalty
            curriculum_difficulty="easy"
        )

    @staticmethod
    def fast_training() -> HRGEnvironmentConfig:
        """Ultra-fast configuration for quick training iterations"""
        return HRGEnvironmentConfig(
            grid_size=6,
            max_steps=100,
            num_obstacles=0,
            num_gold=1,
            num_wood=5,
            gold_cluster_center=(4, 4),
            gold_respawn_time=25,  # Faster respawn
            wood_respawn_time=25,  # Faster respawn
            scout_vision_range=3,  # Reduced for performance
            worker_vision_range=2,  # Reduced for performance
            transporter_vision_range=3,  # Reduced for performance
            worker_carry_capacity=5,  # Increased capacity for faster completion
            transporter_carry_capacity=10,  # Increased capacity
            worker_gather_time=1,  # Faster gathering
            gather_reward_ratio=0.3,  # Higher immediate rewards
            transfer_reward_ratio=0.1,  # Higher transfer rewards
            deposit_reward_ratio=0.7,  # Higher deposit rewards
            step_penalty=0.001,  # Almost no time penalty
            render_mode=None,  # Disable rendering for performance
            curriculum_difficulty="easy"
        )

    @staticmethod
    def ultra_fast() -> HRGEnvironmentConfig:
        """Ultra-ultra fast configuration with minimal agents"""
        return HRGEnvironmentConfig(
            grid_size=6,
            max_steps=80,
            num_obstacles=2,
            num_gold=1,
            num_wood=4,
            # Minimal agent configuration
            agent_config={
                'scouts': [],  # No scouts
                'workers': [AgentType.WORKER],  # Only 1 worker
                'transporters': [AgentType.TRANSPORTER]  # Only 1 transporter
            },
            # Enhanced vision for better learning with fewer agents
            worker_vision_range=2,
            transporter_vision_range=3,
            # High capacity for efficiency
            worker_carry_capacity=5,
            transporter_carry_capacity=10,
            # Fast gathering
            worker_gather_time=1,
            # High rewards for speed
            gather_reward_ratio=0.4,
            transfer_reward_ratio=0.2,
            deposit_reward_ratio=0.8,
            # Minimal penalties
            step_penalty=0.001,
            invalid_move_penalty=0.05,
            # Fast respawn
            gold_respawn_time=20,
            wood_respawn_time=20,
            # Disable rendering
            render_mode=None,
            curriculum_difficulty="easy"
        )

    @staticmethod
    def normal() -> HRGEnvironmentConfig:
        """Normal configuration for standard evaluation"""
        return HRGEnvironmentConfig(
            curriculum_difficulty="normal"
        )

    @staticmethod
    def hard() -> HRGEnvironmentConfig:
        """Hard configuration for challenging scenarios"""
        return HRGEnvironmentConfig(
            grid_size=12,
            max_steps=150,
            num_obstacles=20,
            num_gold=4,
            num_wood=8,
            gold_cluster_radius=3,
            gold_cluster_center=(9, 9),
            scout_vision_range=4,  # Reduced vision
            worker_carry_capacity=1,  # Reduced capacity
            worker_energy_consumption_gather=0.12,  # Higher energy cost
            transporter_energy_consumption_move=0.05,
            gather_reward_ratio=0.05,  # Lower immediate rewards
            step_penalty=0.02,  # Higher time penalty
            invalid_move_penalty=0.2,
            curriculum_difficulty="hard"
        )

    @staticmethod
    def communication_focused() -> HRGEnvironmentConfig:
        """Configuration focused on testing communication"""
        return HRGEnvironmentConfig(
            grid_size=10,
            max_steps=250,
            num_obstacles=15,  # More obstacles to create information asymmetry
            num_gold=3,
            num_wood=8,
            scout_vision_range=6,  # Extended vision for scouts
            worker_vision_range=2,  # Limited vision for workers
            transporter_vision_range=3,
            enable_communication=True,
            communication_cost=0.02,
            max_message_history=5,
            message_dim=32,
            curriculum_difficulty="normal"
        )

    @staticmethod
    def coordination_focused() -> HRGEnvironmentConfig:
        """Configuration focused on testing agent coordination"""
        return HRGEnvironmentConfig(
            grid_size=10,
            max_steps=200,
            num_obstacles=8,
            num_gold=5,  # More gold to create competition
            num_wood=12,
            worker_carry_capacity=1,  # Limited capacity forces coordination
            transporter_carry_capacity=3,  # Moderate transporter capacity
            gather_reward_ratio=0.15,
            transfer_reward_ratio=0.1,  # Higher transfer rewards
            deposit_reward_ratio=0.4,
            curriculum_difficulty="normal"
        )

    @staticmethod
    def exploration_focused() -> HRGEnvironmentConfig:
        """Configuration focused on testing exploration strategies"""
        return HRGEnvironmentConfig(
            grid_size=15,  # Larger map
            max_steps=400,
            num_obstacles=25,
            num_gold=2,
            num_wood=20,  # More scattered resources
            gold_cluster_center=(12, 12),
            gold_cluster_radius=1,  # Tight gold clusters
            wood_min_distance=3,  # More spread out wood
            scout_vision_range=4,
            worker_vision_range=3,
            gather_reward_ratio=0.08,
            resource_diversity_bonus=0.2,
            wood_diversity_bonus=0.1,
            curriculum_difficulty="hard"
        )

    @staticmethod
    def role_specialization_test() -> HRGEnvironmentConfig:
        """Configuration to test role specialization effectiveness"""
        return HRGEnvironmentConfig(
            grid_size=10,
            max_steps=200,
            num_obstacles=12,
            num_gold=3,
            num_wood=10,
            # Enhanced scout abilities
            scout_vision_range=6,
            scout_move_speed=2.5,
            scout_energy_consumption=0.03,
            # Standard worker abilities
            worker_vision_range=3,
            worker_carry_capacity=2,
            worker_gather_time=2,
            # Enhanced transporter abilities
            transporter_vision_range=5,
            transporter_move_speed=2.0,
            transporter_carry_capacity=6,
            # High rewards for role-specific actions
            transfer_reward_ratio=0.15,
            deposit_reward_ratio=0.6,
            curriculum_difficulty="normal"
        )


class HRGExperimentConfigurations:
    """Configuration sets for different experimental protocols"""

    @staticmethod
    def get_ablation_configs() -> Dict[str, HRGEnvironmentConfig]:
        """Get configurations for ablation studies"""
        return {
            'baseline': HRGPresetConfigs.normal(),
            'no_vision_advantage': HRGEnvironmentConfig(
                scout_vision_range=3,
                worker_vision_range=3,
                transporter_vision_range=3
            ),
            'no_speed_advantage': HRGEnvironmentConfig(
                scout_move_speed=1.0,
                worker_move_speed=1.0,
                transporter_move_speed=1.0
            ),
            'no_carry_advantage': HRGEnvironmentConfig(
                worker_carry_capacity=1,
                transporter_carry_capacity=1
            ),
            'high_communication_cost': HRGEnvironmentConfig(
                enable_communication=True,
                communication_cost=0.1
            ),
            'no_communication': HRGEnvironmentConfig(
                enable_communication=False
            )
        }

    @staticmethod
    def get_curriculum_configs() -> List[HRGEnvironmentConfig]:
        """Get configurations for curriculum learning"""
        return [
            HRGPresetConfigs.easy(),      # Stage 1
            HRGPresetConfigs.normal(),    # Stage 2
            HRGPresetConfigs.hard(),      # Stage 3
            HRGPresetConfigs.exploration_focused(),  # Stage 4
        ]

    @staticmethod
    def get_comparison_configs() -> Dict[str, HRGEnvironmentConfig]:
        """Get configurations for comparing with other environments"""
        return {
            'hrg_standard': HRGPresetConfigs.normal(),
            'hrg_easy': HRGPresetConfigs.easy(),
            'hrg_hard': HRGPresetConfigs.hard(),
            'hrg_communication': HRGPresetConfigs.communication_focused(),
            'hrg_coordination': HRGPresetConfigs.coordination_focused(),
        }


def get_config_by_name(config_name: str, **kwargs) -> HRGEnvironmentConfig:
    """
    Get configuration by name with optional overrides

    Args:
        config_name: Name of the configuration
        **kwargs: Parameters to override

    Returns:
        HRGEnvironmentConfig: The requested configuration
    """
    config_map = {
        'easy': HRGPresetConfigs.easy,
        'normal': HRGPresetConfigs.normal,
        'hard': HRGPresetConfigs.hard,
        'communication': HRGPresetConfigs.communication_focused,
        'coordination': HRGPresetConfigs.coordination_focused,
        'exploration': HRGPresetConfigs.exploration_focused,
        'role_test': HRGPresetConfigs.role_specialization_test,
        'ultra_fast': HRGPresetConfigs.ultra_fast,
    }

    if config_name not in config_map:
        raise ValueError(f"Unknown configuration: {config_name}. "
                        f"Available: {list(config_map.keys())}")

    config = config_map[config_name]()

    # Apply any parameter overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"Warning: Unknown configuration parameter: {key}")

    return config


def save_config(config: HRGEnvironmentConfig, filepath: str):
    """Save configuration to file"""
    import json
    with open(filepath, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)


def load_config(filepath: str) -> HRGEnvironmentConfig:
    """Load configuration from file"""
    import json
    with open(filepath, 'r') as f:
        config_dict = json.load(f)
    return HRGEnvironmentConfig.from_dict(config_dict)