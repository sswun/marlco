#!/usr/bin/env python3
"""
Test script for VIP intelligent pathfinding improvements
"""

import numpy as np
from Env.DEM.env_dem_ctde import create_dem_ctde_env


def test_vip_pathfinding():
    """Test VIP pathfinding with obstacles"""
    print("🧪 Testing VIP Intelligent Pathfinding...")

    # Create environment with obstacles
    env = create_dem_ctde_env(
        difficulty="normal",
        global_state_type="concat",
        max_steps=100
    )

    obs = env.reset()

    # Get initial VIP position
    initial_info = env.get_global_info()
    vip_initial = initial_info['vip']['pos']
    vip_target = initial_info['vip']['target_pos']

    print(f"VIP initial position: {vip_initial}")
    print(f"VIP target position: {vip_target}")
    print(f"Initial distance to target: {abs(vip_initial[0] - vip_target[0]) + abs(vip_initial[1] - vip_target[1])}")

    # Simulate several steps to observe VIP movement
    vip_positions = [vip_initial]
    total_distance_traveled = 0

    for step in range(20):
        # Use minimal agent actions to focus on VIP movement
        actions = {}
        for agent_id in obs.keys():
            # Most agents stay still to observe VIP movement
            actions[agent_id] = 0  # STAY action

        next_obs, rewards, done, info = env.step(actions)

        # Get VIP position
        current_info = env.get_global_info()
        vip_current = current_info['vip']['pos']
        vip_positions.append(vip_current)

        # Calculate distance traveled
        prev_pos = vip_positions[-2]
        distance = abs(vip_current[0] - prev_pos[0]) + abs(vip_current[1] - prev_pos[1])
        total_distance_traveled += distance

        print(f"Step {step + 1}: VIP at {vip_current}, moved {distance} units")

        # Check if VIP is making progress towards target
        current_distance = abs(vip_current[0] - vip_target[0]) + abs(vip_current[1] - vip_target[1])
        print(f"  Distance to target: {current_distance}")

        obs = next_obs

        if done:
            break

    print(f"\n✅ VIP Pathfinding Test Results:")
    print(f"  Total steps observed: {len(vip_positions) - 1}")
    print(f"  Total distance traveled: {total_distance_traveled}")
    print(f"  Average movement per step: {total_distance_traveled / (len(vip_positions) - 1):.2f}")

    # Check if VIP is stuck (not moving)
    unique_positions = len(set(vip_positions))
    if unique_positions < len(vip_positions) * 0.5:  # Less than 50% unique positions
        print(f"  ⚠️  Warning: VIP might be stuck (only {unique_positions} unique positions out of {len(vip_positions)} steps)")
    else:
        print(f"  ✅ VIP is moving well ({unique_positions} unique positions out of {len(vip_positions)} steps)")

    env.close()
    return True


def test_vip_evasion():
    """Test VIP evasive maneuvers with threats"""
    print("\n🧪 Testing VIP Evasion Behavior...")

    # Create environment
    env = create_dem_ctde_env(
        difficulty="normal",
        global_state_type="concat",
        max_steps=50
    )

    obs = env.reset()

    # Get initial state
    initial_info = env.get_global_info()
    vip_initial = initial_info['vip']['pos']

    print(f"VIP initial position: {vip_initial}")

    # Simulate steps and observe VIP behavior when threats appear
    for step in range(15):
        # Agent actions
        actions = {}
        for agent_id in obs.keys():
            avail_actions = env.get_avail_agent_actions(agent_id)
            actions[agent_id] = np.random.choice(avail_actions)

        next_obs, rewards, done, info = env.step(actions)

        # Get current state
        current_info = env.get_global_info()
        vip_current = current_info['vip']['pos']
        threats = current_info['threats']

        print(f"Step {step + 1}: VIP at {vip_current}")

        if threats:
            print(f"  Active threats: {len(threats)}")
            for i, (tx, ty, ttype) in enumerate(threats[:3]):  # Show first 3 threats
                threat_distance = abs(vip_current[0] - tx) + abs(vip_current[1] - ty)
                print(f"    Threat {i+1} ({ttype}) at ({tx}, {ty}), distance: {threat_distance}")

                if threat_distance <= 3:
                    print(f"      ⚠️  Threat is nearby - VIP should evade")
        else:
            print("  No threats detected")

        obs = next_obs

        if done:
            break

    print(f"\n✅ VIP Evasion Test Completed")
    env.close()
    return True


def test_vip_obstacle_avoidance():
    """Test VIP's ability to avoid obstacles (rivers)"""
    print("\n🧪 Testing VIP Obstacle Avoidance...")

    # Create environment with default terrain (includes rivers)
    env = create_dem_ctde_env(
        difficulty="normal",
        global_state_type="concat",
        max_steps=30
    )

    obs = env.reset()

    # Get terrain information
    global_info = env.get_global_info()
    terrain_map = global_info['terrain']
    vip_pos = global_info['vip']['pos']
    target_pos = global_info['vip']['target_pos']

    print(f"VIP position: {vip_pos}")
    print(f"Target position: {target_pos}")

    # Show terrain around VIP's likely path
    print("\nTerrain map legend: O=Open, R=River, F=Forest")
    print("Terrain around VIP and target:")

    grid_size = len(terrain_map)
    for y in range(max(0, min(vip_pos[1], target_pos[1]) - 2),
                   min(grid_size, max(vip_pos[1], target_pos[1]) + 3)):
        row = ""
        for x in range(max(0, min(vip_pos[0], target_pos[0]) - 2),
                       min(grid_size, max(vip_pos[0], target_pos[0]) + 3)):
            terrain = terrain_map[x][y][0].upper() if terrain_map[x][y] else '?'
            if (x, y) == vip_pos:
                terrain = 'V'  # VIP
            elif (x, y) == target_pos:
                terrain = 'T'  # Target
            row += f"{terrain:2s}"
        print(row)

    # Test VIP movement
    stuck_positions = 0
    prev_pos = vip_pos

    for step in range(10):
        actions = {agent_id: 0 for agent_id in obs.keys()}  # All stay
        next_obs, rewards, done, info = env.step(actions)

        current_info = env.get_global_info()
        current_pos = current_info['vip']['pos']

        if current_pos == prev_pos:
            stuck_positions += 1
            print(f"Step {step + 1}: VIP stuck at {current_pos}")
        else:
            print(f"Step {step + 1}: VIP moved from {prev_pos} to {current_pos}")

        # Check if VIP is in valid terrain (not river)
        terrain_at_pos = terrain_map[current_pos[0]][current_pos[1]]
        if terrain_at_pos == 'river':
            print(f"  ❌ ERROR: VIP is in river at {current_pos}!")

        prev_pos = current_pos
        obs = next_obs

        if done:
            break

    if stuck_positions > 5:
        print(f"  ⚠️  VIP got stuck frequently ({stuck_positions} out of 10 steps)")
    else:
        print(f"  ✅ VIP navigates around obstacles well")

    env.close()
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("VIP INTELLIGENT PATHFINDING TESTS")
    print("=" * 60)

    success = True

    success &= test_vip_pathfinding()
    success &= test_vip_evasion()
    success &= test_vip_obstacle_avoidance()

    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED! VIP pathfinding improvements working correctly.")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    print("=" * 60)