"""
SMAC Configuration Module

Configuration classes and predefined settings for different SMAC scenarios.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class SMACConfig:
    """Configuration class for SMAC environment wrapper"""

    # Scenario configuration
    map_name: str = "8m"

    # Difficulty settings
    difficulty: str = "normal"  # "easy", "normal", "hard"

    # Episode settings
    episode_limit: Optional[int] = None

    # Rendering
    render_mode: Optional[str] = None  # "human", "rgb_array", or None

    # Debug settings
    debug: bool = False
    seed: Optional[int] = None

    def __post_init__(self):
        """Post-initialization setup"""
        # Set episode limit based on map if not specified
        if self.episode_limit is None:
            self.episode_limit = self._get_default_episode_limit()

    def _get_default_episode_limit(self) -> int:
        """Get default episode limit based on map"""
        episode_limits = {
            "3m": 120, "8m": 120, "2m_vs_1z": 120,
            "2s3z": 120, "3s5z": 120, "1c3s5z": 120,
            "3s": 120, "1s3z": 120, "2s_vs_1sc": 120,
            "MMM2": 120, "corridor": 200, "6h": 200,
            "3s5z_vs_3s6z": 180, "3s_vs_5z": 200,
            "bane_vs_bane": 200, "MMM": 180, "2c_vs_64zg": 200,
            "27m_vs_30m": 150, "5m_vs_6m": 150, "8m_vs_9m": 150,
            "2s_vs_1sc": 120, "3s_vs_3z": 120, "3s5z": 120,
            "1c3s5z": 120, "2s3z": 120, "3s_vs_5z": 200,
            "1s3z": 120, "10m_vs_11m": 150, "2s_vs_1sc": 120,
            "3s5z_vs_3s6z": 180, "3s_vs_3z": 120, "3s_vs_5z": 200,
            "6h_vs_8z": 200, "2m_vs_1z": 120, "3s5z": 120,
            "3s_vs_3z": 120, "1s3z": 120, "3s_vs_5z": 200,
            "1c3s5z": 120, "2s3z": 120, "3s_vs_3z": 120,
            "3s_vs_5z": 200, "1s3z": 120, "3s5z_vs_3s6z": 180
        }
        return episode_limits.get(self.map_name, 120)


def get_config_by_name(config_name: str) -> SMACConfig:
    """Get predefined configuration by name"""

    configs = {
        "easy": SMACConfig(
            map_name="8m",
            difficulty="easy",
            episode_limit=200,
            debug=False
        ),

        "normal": SMACConfig(
            map_name="8m",
            difficulty="normal",
            episode_limit=120,
            debug=False
        ),

        "hard": SMACConfig(
            map_name="6h",
            difficulty="hard",
            episode_limit=200,
            debug=False
        ),

        "debug": SMACConfig(
            map_name="8m",
            difficulty="debug",
            episode_limit=50,
            debug=True,
            seed=42
        ),

        "corridor": SMACConfig(
            map_name="corridor",
            difficulty="normal",
            episode_limit=200,
            debug=False
        ),

        "6h": SMACConfig(
            map_name="6h",
            difficulty="hard",
            episode_limit=200,
            debug=False
        ),

        "MMM": SMACConfig(
            map_name="MMM",
            difficulty="normal",
            episode_limit=180,
            debug=False
        ),

        "MMM2": SMACConfig(
            map_name="MMM2",
            difficulty="normal",
            episode_limit=120,
            debug=False
        ),

        "2s_vs_1sc": SMACConfig(
            map_name="2s_vs_1sc",
            difficulty="normal",
            episode_limit=120,
            debug=False
        ),

        "3s5z": SMACConfig(
            map_name="3s5z",
            difficulty="normal",
            episode_limit=120,
            debug=False
        ),

        "bane_vs_bane": SMACConfig(
            map_name="bane_vs_bane",
            difficulty="normal",
            episode_limit=200,
            debug=False
        ),

        "2c_vs_64zg": SMACConfig(
            map_name="2c_vs_64zg",
            difficulty="hard",
            episode_limit=200,
            debug=False
        )
    }

    if config_name not in configs:
        raise ValueError(f"Unknown configuration: {config_name}. "
                        f"Available configs: {list(configs.keys())}")

    return configs[config_name]


def get_debug_config() -> SMACConfig:
    """Get debug configuration for quick testing"""
    return get_config_by_name("debug")


def get_easy_config() -> SMACConfig:
    """Get easy configuration"""
    return get_config_by_name("easy")


def get_normal_config() -> SMACConfig:
    """Get normal configuration"""
    return get_config_by_name("normal")


def get_hard_config() -> SMACConfig:
    """Get hard configuration"""
    return get_config_by_name("hard")


def create_custom_config(
    map_name: str = "8m",
    difficulty: str = "normal",
    episode_limit: int = None,
    **kwargs
) -> SMACConfig:
    """Create custom configuration"""

    return SMACConfig(
        map_name=map_name,
        difficulty=difficulty,
        episode_limit=episode_limit,
        **kwargs
    )