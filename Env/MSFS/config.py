"""
MSFS Environment Configuration Module

This module provides configuration classes and preset configurations
for different experimental setups in the MSFS environment.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class MSFSConfig:
    """Comprehensive configuration for MSFS environment"""

    # Environment settings
    max_steps: int = 50
    num_agents: int = 2
    seed: Optional[int] = None

    # Order settings
    queue_limit: int = 20
    simple_order_value: float = 5.0
    complex_order_value: float = 10.0

    # Agent settings
    move_cooldown_time: int = 1

    # Enhanced reward settings for better exploration
    # Action-based rewards (immediate feedback)
    move_toward_target: float = 0.1
    pickup_material: float = 0.2
    start_processing: float = 0.3
    complete_stage: float = 0.5
    deliver_order: float = 1.0

    # Progress rewards (milestone-based)
    raw_completion: float = 1.0
    assembly_completion: float = 2.0
    packaging_completion: float = 3.0
    order_delivery: float = 5.0
    smooth_transition: float = 0.5
    no_queue_bonus: float = 0.3

    # Cooperation rewards (team-based)
    successful_handoff: float = 0.8
    workstation_ready: float = 0.4
    concurrent_processing: float = 0.6
    balanced_workload: float = 0.3

    # Role emergence rewards
    collector_focus: float = 0.2
    processor_focus: float = 0.3
    packager_focus: float = 0.4
    stick_to_role: float = 0.1
    switch_when_needed: float = 0.5

    # Reduced penalties for exploration
    step_penalty: float = 0.0  # Removed time pressure
    idle_penalty: float = 0.0   # Allow strategic waiting
    invalid_action_penalty: float = -0.1  # Much lighter penalty

    # Legacy settings (kept for compatibility)
    specialization_reward: float = 0.5
    finishing_reward: float = 1.0
    specialization_threshold: int = 3  # Consecutive processes for specialization reward
    finishing_phase_start: int = 35

    # Visualization settings
    render_mode: str = "rgb_array"  # "human", "rgb_array", or None
    render_fps: int = 4
    render_grid_size: int = 800
    show_agent_ids: bool = True
    show_queue_info: bool = True

    # Learning settings
    difficulty: str = "normal"  # "easy", "normal", "hard"
    curriculum_stage: int = 1

    # Role emergence settings
    enable_role_emergence_rewards: bool = True
    role_switch_penalty: float = 0.1

    # Performance settings
    track_utilization: bool = True
    track_specialization: bool = True

    def validate(self) -> None:
        """Validate configuration parameters"""
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.num_agents <= 0 or self.num_agents > 3:
            raise ValueError("num_agents must be between 1 and 3")
        if self.queue_limit <= 0:
            raise ValueError("queue_limit must be positive")
        if self.simple_order_value <= 0 or self.complex_order_value <= 0:
            raise ValueError("order values must be positive")
        if self.specialization_reward < 0 or self.finishing_reward < 0:
            raise ValueError("rewards must be non-negative")
        if self.specialization_threshold <= 0:
            raise ValueError("specialization_threshold must be positive")
        if self.finishing_phase_start < 0:
            raise ValueError("finishing_phase_start must be non-negative")
        # Validate new reward parameters are non-negative (except penalties)
        reward_attrs = [
            'move_toward_target', 'pickup_material', 'start_processing', 'complete_stage', 'deliver_order',
            'raw_completion', 'assembly_completion', 'packaging_completion', 'order_delivery',
            'smooth_transition', 'no_queue_bonus', 'successful_handoff', 'workstation_ready',
            'concurrent_processing', 'balanced_workload', 'collector_focus', 'processor_focus',
            'packager_focus', 'stick_to_role', 'switch_when_needed'
        ]
        for attr in reward_attrs:
            if getattr(self, attr) < 0:
                raise ValueError(f"{attr} must be non-negative")
        # Allow finishing_phase_start to be >= max_steps, just adjust during runtime
        if self.difficulty not in ["easy", "normal", "hard"]:
            raise ValueError("difficulty must be one of 'easy', 'normal', 'hard'")
        if self.curriculum_stage <= 0:
            raise ValueError("curriculum_stage must be positive")

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'max_steps': self.max_steps,
            'num_agents': self.num_agents,
            'seed': self.seed,
            'queue_limit': self.queue_limit,
            'simple_order_value': self.simple_order_value,
            'complex_order_value': self.complex_order_value,
            'move_cooldown_time': self.move_cooldown_time,
            # Enhanced reward settings
            'move_toward_target': self.move_toward_target,
            'pickup_material': self.pickup_material,
            'start_processing': self.start_processing,
            'complete_stage': self.complete_stage,
            'deliver_order': self.deliver_order,
            'raw_completion': self.raw_completion,
            'assembly_completion': self.assembly_completion,
            'packaging_completion': self.packaging_completion,
            'order_delivery': self.order_delivery,
            'smooth_transition': self.smooth_transition,
            'no_queue_bonus': self.no_queue_bonus,
            'successful_handoff': self.successful_handoff,
            'workstation_ready': self.workstation_ready,
            'concurrent_processing': self.concurrent_processing,
            'balanced_workload': self.balanced_workload,
            'collector_focus': self.collector_focus,
            'processor_focus': self.processor_focus,
            'packager_focus': self.packager_focus,
            'stick_to_role': self.stick_to_role,
            'switch_when_needed': self.switch_when_needed,
            # Penalties
            'step_penalty': self.step_penalty,
            'idle_penalty': self.idle_penalty,
            'invalid_action_penalty': self.invalid_action_penalty,
            # Legacy settings
            'specialization_reward': self.specialization_reward,
            'finishing_reward': self.finishing_reward,
            'specialization_threshold': self.specialization_threshold,
            'finishing_phase_start': self.finishing_phase_start,
            # Other settings
            'render_mode': self.render_mode,
            'render_fps': self.render_fps,
            'render_grid_size': self.render_grid_size,
            'show_agent_ids': self.show_agent_ids,
            'show_queue_info': self.show_queue_info,
            'difficulty': self.difficulty,
            'curriculum_stage': self.curriculum_stage,
            'enable_role_emergence_rewards': self.enable_role_emergence_rewards,
            'role_switch_penalty': self.role_switch_penalty,
            'track_utilization': self.track_utilization,
            'track_specialization': self.track_specialization
        }


class MSFSPresetConfigs:
    """Preset configurations for different experimental scenarios"""

    @staticmethod
    def easy() -> MSFSConfig:
        """Easy configuration for initial training with enhanced exploration rewards"""
        return MSFSConfig(
            max_steps=60,
            num_agents=2,
            simple_order_value=7.0,  # Higher rewards
            complex_order_value=12.0,
            # Enhanced exploration rewards
            move_toward_target=0.15,  # Higher reward for moving correctly
            pickup_material=0.3,
            start_processing=0.4,
            complete_stage=0.7,
            deliver_order=1.5,
            raw_completion=1.5,
            assembly_completion=2.5,
            packaging_completion=3.5,
            order_delivery=7.0,  # High reward for completion
            successful_handoff=1.0,
            collector_focus=0.3,
            processor_focus=0.4,
            packager_focus=0.5,
            # Reduced penalties
            step_penalty=0.0,
            idle_penalty=0.0,
            invalid_action_penalty=-0.05,  # Very light penalty
            # Legacy settings
            specialization_reward=1.0,  # Higher role rewards
            finishing_reward=1.5,
            specialization_threshold=2,  # Easier to specialize
            difficulty="easy",
            curriculum_stage=1
        )

    @staticmethod
    def normal() -> MSFSConfig:
        """Normal configuration for standard evaluation"""
        return MSFSConfig(
            difficulty="normal",
            curriculum_stage=2
        )

    @staticmethod
    def hard() -> MSFSConfig:
        """Hard configuration for challenging scenarios"""
        return MSFSConfig(
            max_steps=40,  # Shorter episodes
            num_agents=2,
            simple_order_value=4.0,  # Lower rewards
            complex_order_value=8.0,
            step_penalty=0.02,  # Higher time penalty
            idle_penalty=0.01,
            specialization_reward=0.3,  # Lower role rewards
            finishing_reward=0.8,
            specialization_threshold=4,  # Harder to specialize
            role_switch_penalty=0.2,  # Higher penalty for switching
            difficulty="hard",
            curriculum_stage=3
        )

    @staticmethod
    def single_agent() -> MSFSConfig:
        """Single agent configuration for basic learning"""
        return MSFSConfig(
            max_steps=50,
            num_agents=1,
            simple_order_value=6.0,
            complex_order_value=11.0,
            step_penalty=0.008,
            specialization_reward=0.8,
            specialization_threshold=2,
            difficulty="easy",
            curriculum_stage=1
        )

    @staticmethod
    def three_agent() -> MSFSConfig:
        """Three agent configuration for complex coordination"""
        return MSFSConfig(
            max_steps=50,
            num_agents=3,
            step_penalty=0.012,  # Slightly higher penalty for more agents
            specialization_reward=0.4,
            role_switch_penalty=0.15,
            difficulty="normal",
            curriculum_stage=3
        )

    @staticmethod
    def role_emergence_focus() -> MSFSConfig:
        """Configuration focused on role emergence"""
        return MSFSConfig(
            max_steps=60,
            num_agents=2,
            specialization_reward=1.5,  # High specialization rewards
            finishing_reward=2.0,
            role_switch_penalty=0.3,  # High penalty for switching
            specialization_threshold=3,
            enable_role_emergence_rewards=True,
            track_specialization=True,
            difficulty="normal",
            curriculum_stage=2
        )

    @staticmethod
    def efficiency_focus() -> MSFSConfig:
        """Configuration focused on efficiency"""
        return MSFSConfig(
            max_steps=40,
            num_agents=2,
            step_penalty=0.03,  # High time pressure
            idle_penalty=0.02,
            simple_order_value=4.0,
            complex_order_value=7.0,
            specialization_reward=0.2,  # Lower role rewards
            enable_role_emergence_rewards=False,
            difficulty="hard",
            curriculum_stage=3
        )

    @staticmethod
    def curriculum_stage1() -> MSFSConfig:
        """Stage 1: Basic single agent, simple orders"""
        return MSFSConfig(
            max_steps=50,
            num_agents=1,
            queue_limit=10,
            simple_order_value=8.0,
            complex_order_value=0.0,  # No complex orders initially
            step_penalty=0.005,
            enable_role_emergence_rewards=False,
            difficulty="easy",
            curriculum_stage=1
        )

    @staticmethod
    def curriculum_stage2() -> MSFSConfig:
        """Stage 2: Two agents, mixed orders"""
        return MSFSConfig(
            max_steps=50,
            num_agents=2,
            simple_order_value=6.0,
            complex_order_value=10.0,
            step_penalty=0.01,
            enable_role_emergence_rewards=True,
            specialization_reward=0.5,
            difficulty="normal",
            curriculum_stage=2
        )

    @staticmethod
    def curriculum_stage3() -> MSFSConfig:
        """Stage 3: Full complexity with role rewards"""
        return MSFSConfig(
            max_steps=50,
            num_agents=2,
            simple_order_value=5.0,
            complex_order_value=10.0,
            step_penalty=0.01,
            idle_penalty=0.005,
            specialization_reward=0.5,
            finishing_reward=1.0,
            role_switch_penalty=0.1,
            enable_role_emergence_rewards=True,
            track_specialization=True,
            difficulty="normal",
            curriculum_stage=3
        )


class MSFSExperimentConfigurations:
    """Configuration sets for different experimental protocols"""

    @staticmethod
    def get_ablation_configs() -> Dict[str, MSFSConfig]:
        """Get configurations for ablation studies"""
        return {
            'baseline': MSFSPresetConfigs.normal(),
            'no_role_rewards': MSFSConfig(
                enable_role_emergence_rewards=False,
                specialization_reward=0.0,
                finishing_reward=0.0
            ),
            'high_specialization': MSFSConfig(
                specialization_reward=1.0,
                specialization_threshold=2,
                role_switch_penalty=0.2
            ),
            'low_specialization': MSFSConfig(
                specialization_reward=0.2,
                specialization_threshold=5,
                role_switch_penalty=0.05
            ),
            'single_agent': MSFSPresetConfigs.single_agent(),
            'three_agent': MSFSPresetConfigs.three_agent()
        }

    @staticmethod
    def get_curriculum_configs() -> List[MSFSConfig]:
        """Get configurations for curriculum learning"""
        return [
            MSFSPresetConfigs.curriculum_stage1(),
            MSFSPresetConfigs.curriculum_stage2(),
            MSFSPresetConfigs.curriculum_stage3()
        ]

    @staticmethod
    def get_comparison_configs() -> Dict[str, MSFSConfig]:
        """Get configurations for comparing with other environments"""
        return {
            'msfs_easy': MSFSPresetConfigs.easy(),
            'msfs_normal': MSFSPresetConfigs.normal(),
            'msfs_hard': MSFSPresetConfigs.hard(),
            'msfs_role_focus': MSFSPresetConfigs.role_emergence_focus(),
            'msfs_efficiency': MSFSPresetConfigs.efficiency_focus()
        }


def get_config_by_name(config_name: str, **kwargs) -> MSFSConfig:
    """
    Get configuration by name with optional overrides

    Args:
        config_name: Name of the configuration
        **kwargs: Parameters to override

    Returns:
        MSFSConfig: The requested configuration
    """
    config_map = {
        'easy': MSFSPresetConfigs.easy,
        'normal': MSFSPresetConfigs.normal,
        'hard': MSFSPresetConfigs.hard,
        'single_agent': MSFSPresetConfigs.single_agent,
        'three_agent': MSFSPresetConfigs.three_agent,
        'role_emergence': MSFSPresetConfigs.role_emergence_focus,
        'efficiency': MSFSPresetConfigs.efficiency_focus,
        'curriculum1': MSFSPresetConfigs.curriculum_stage1,
        'curriculum2': MSFSPresetConfigs.curriculum_stage2,
        'curriculum3': MSFSPresetConfigs.curriculum_stage3
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


def save_config(config: MSFSConfig, filepath: str):
    """Save configuration to file"""
    import json
    with open(filepath, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)


def load_config(filepath: str) -> MSFSConfig:
    """Load configuration from file"""
    import json
    with open(filepath, 'r') as f:
        config_dict = json.load(f)
    return MSFSConfig(**config_dict)