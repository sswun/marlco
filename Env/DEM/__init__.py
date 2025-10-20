"""
DEM (Dynamic Escort Mission) Environment Package

This package implements a multi-agent reinforcement learning environment
for dynamic role-based VIP escort missions with emergent role formation.
"""

from .env_dem import DEMEnv, DEMConfig
from .env_dem_ctde import DEMCTDEWrapper, create_dem_ctde_env
from .env_dem import create_dem_env

__all__ = [
    'DEMEnv',
    'DEMConfig',
    'DEMCTDEWrapper',
    'create_dem_ctde_env',
    'create_dem_env'
]

__version__ = "1.0.0"