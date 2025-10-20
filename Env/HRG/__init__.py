"""
HRG (Heterogeneous Resource Gathering) Environment Package

This package provides the HRG environment for multi-agent reinforcement learning.
"""

from .core import (
    Agent, AgentType, ResourceType, ActionType, Position, Resource,
    AGENT_CONFIGS, RESOURCE_CONFIGS
)
from .config import (
    HRGEnvironmentConfig, HRGPresetConfigs, get_config_by_name
)
from .env_hrg import HRGEnv
from .env_hrg_ctde import HRGCTDEWrapper as HRGCTDEEnv
from .env_hrg_fast import HRGFastEnv, create_hrg_fast_env
from .env_hrg_fast_ctde import HRGFastCTDEEnv, create_hrg_fast_ctde_env
from .env_hrg_ultra_fast import HRGUltraFastEnv, create_hrg_ultra_fast_env
from .env_hrg_ultra_fast_ctde import HRGUltraFastCTDEEnv, create_hrg_ultra_fast_ctde_env


def create_hrg_env(difficulty: str = "normal", **kwargs):
    """Create HRG environment based on difficulty"""
    config = get_config_by_name(difficulty, **kwargs)
    return HRGEnv(config)


def create_hrg_ctde_env(difficulty: str = "easy_ctde", **kwargs):
    """Create HRG CTDE environment based on difficulty"""
    config = get_config_by_name(difficulty, **kwargs)
    return HRGCTDEEnv(config)


__all__ = [
    # Core classes
    'Agent', 'AgentType', 'ResourceType', 'ActionType', 'Position', 'Resource',
    'AGENT_CONFIGS', 'RESOURCE_CONFIGS',

    # Configuration
    'HRGEnvironmentConfig', 'HRGPresetConfigs', 'get_config_by_name',

    # Environment variants
    'HRGEnv', 'HRGCTDEEnv',
    'HRGFastEnv', 'HRGFastCTDEEnv',
    'HRGUltraFastEnv', 'HRGUltraFastCTDEEnv',

    # Factory functions
    'create_hrg_env', 'create_hrg_ctde_env',
    'create_hrg_fast_env', 'create_hrg_fast_ctde_env',
    'create_hrg_ultra_fast_env', 'create_hrg_ultra_fast_ctde_env',
]

__version__ = "1.1.0"