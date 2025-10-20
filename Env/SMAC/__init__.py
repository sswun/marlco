"""
SMAC (StarCraft Multi-Agent Challenge) Environment Wrapper

A wrapper for the SMAC library that provides compatibility with the
unified environment interface used by DEM, HRG, and MSFS environments.
"""

from .env_smac import SMACEnv, create_smac_env, create_smac_env_easy, create_smac_env_normal, create_smac_env_hard
from .env_smac_ctde import SMACCTDEEnv, create_smac_ctde_env
from .config import SMACConfig, get_easy_config, get_normal_config, get_hard_config

__version__ = "1.0.0"
__author__ = "MARL Research Team"

# Export main classes and functions
__all__ = [
    # Environment classes
    "SMACEnv",
    "SMACCTDEEnv",

    # Factory functions
    "create_smac_env",
    "create_smac_ctde_env",
    "create_smac_env_easy",
    "create_smac_env_normal",
    "create_smac_env_hard",

    # Configuration
    "SMACConfig",
    "get_easy_config",
    "get_normal_config",
    "get_hard_config"
]