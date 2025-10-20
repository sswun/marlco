"""
MSFS Environment - Smart Manufacturing Flow Scheduling

A role emergence oriented multi-agent reinforcement learning environment
for intelligent manufacturing flow scheduling.
"""

from .env_msfs import create_msfs_env
from .env_msfs_ctde import create_msfs_ctde_env

__version__ = "1.0.0"
__author__ = "MSFS Environment Team"

__all__ = [
    "create_msfs_env",
    "create_msfs_ctde_env"
]