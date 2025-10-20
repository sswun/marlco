#!/usr/bin/env python3
"""
Debug script to observe VIP pathfinding behavior in detail
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from Env.DEM.env_dem import create_dem_env


def debug_vip_movement():
    """Debug VIP movement with detailed logging"""
    print("🔍 Debug VIP Movement with Detailed Logging...")

    # Create basic environment (non-CTDE) for easier debugging
    env = create_dem_env(difficulty="normal", max_steps=100)

    obs = env.reset()

    # Enable debug logging
    import logging
    logging.basicConfig(level=logging.DEBUG, format='%(message)s')

    print(f"VIP initial: {env.game_state.vip.pos} -> Target: {env.game_state.vip.target_pos}")

    for step in range(20):
        print(f"\n--- Step {step + 1} ---")
        print(f"VIP position: {env.game_state.vip.pos}")
        print(f"VIP position history: {env.vip_position_history}")
        print(f"VIP last progress step: {env.vip_last_progress_step}")

        # Check if stuck detection would trigger
        is_stuck = env._is_vip_stuck(env.game_state.vip.pos, env.game_state.vip.target_pos)
        print(f"Is VIP stuck: {is_stuck}")

        # Let VIP move (agents stay still)
        actions = {}
        for agent_id in obs.keys():
            actions[agent_id] = 0  # STAY

        obs, rewards, done, info = env.step(actions)

        if done:
            break

    print(f"\nFinal VIP position: {env.game_state.vip.pos}")
    print(f"Total unique positions visited: {len(set(env.vip_position_history))}")

    env.close()


if __name__ == "__main__":
    debug_vip_movement()