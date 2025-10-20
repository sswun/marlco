#!/usr/bin/env python3
"""
Test script for improved VIP pathfinding with obstacle avoidance

This script tests the VIP's ability to navigate around obstacles,
particularly the cross-shaped river formation in the center.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from Env.DEM.env_dem_ctde import create_dem_ctde_env


def test_vip_obstacle_navigation():
    """Test VIP navigation around obstacles"""
    print("🧪 Testing VIP Obstacle Navigation...")

    # Create environment with default terrain (includes cross-shaped rivers)
    env = create_dem_ctde_env(
        difficulty="normal",
        global_state_type="concat",
        max_steps=100
    )

    obs = env.reset()

    # Get initial state
    initial_info = env.get_global_info()
    vip_initial = initial_info['vip']['pos']
    vip_target = initial_info['vip']['target_pos']

    print(f"VIP initial position: {vip_initial}")
    print(f"VIP target position: {vip_target}")
    print(f"Initial distance to target: {abs(vip_initial[0] - vip_target[0]) + abs(vip_initial[1] - vip_target[1])}")

    # Show terrain around VIP's path
    terrain_map = initial_info['terrain']
    print("\nTerrain map legend: O=Open, R=River, F=Forest, V=VIP, T=Target")
    print("Terrain overview (showing middle section where obstacles are):")

    grid_size = len(terrain_map)
    center = grid_size // 2
    show_range = 6

    for y in range(max(0, center - show_range), min(grid_size, center + show_range + 1)):
        row = ""
        for x in range(max(0, center - show_range), min(grid_size, center + show_range + 1)):
            terrain = terrain_map[x][y][0].upper() if terrain_map[x][y] else '?'

            # Mark important positions
            if (x, y) == vip_initial:
                terrain = 'V'  # VIP
            elif (x, y) == vip_target:
                terrain = 'T'  # Target
            elif (x, y) == (center, center):
                terrain = 'X' if terrain_map[x][y][0] != 'river' else 'R'  # Center

            row += f"{terrain:2s}"
        print(row)

    # Test VIP movement with intelligent agents
    vip_positions = [vip_initial]
    stuck_periods = []
    total_distance_traveled = 0
    wall_following_used = False

    for step in range(40):
        # Use semi-intelligent agent actions to allow VIP movement
        actions = {}
        for agent_id in obs.keys():
            avail_actions = env.get_avail_agent_actions(agent_id)

            # Simple strategy: agents mostly stay still or move randomly
            # This allows us to focus on VIP movement
            if np.random.random() < 0.7:  # 70% stay still
                actions[agent_id] = 0  # STAY
            else:
                actions[agent_id] = np.random.choice(avail_actions)

        next_obs, rewards, done, info = env.step(actions)

        # Get VIP position
        current_info = env.get_global_info()
        vip_current = current_info['vip']['pos']
        vip_positions.append(vip_current)

        # Calculate distance traveled
        prev_pos = vip_positions[-2]
        distance = abs(vip_current[0] - prev_pos[0]) + abs(vip_current[1] - prev_pos[1])
        total_distance_traveled += distance

        # Check for stuck detection (this would trigger wall-following)
        if len(vip_positions) >= 6:
            recent_positions = vip_positions[-6:]
            unique_positions = len(set(recent_positions))
            if unique_positions <= 3:
                stuck_periods.append(step)
                print(f"Step {step + 1}: VIP appears stuck (repeating {unique_positions} positions)")

        # Check distance to target
        current_distance = abs(vip_current[0] - vip_target[0]) + abs(vip_current[1] - vip_target[1])

        # Provide detailed output for key positions
        if distance == 0 and step > 5:
            print(f"Step {step + 1}: VIP stationary at {vip_current}, distance to target: {current_distance}")
        elif distance > 0:
            print(f"Step {step + 1}: VIP moved from {prev_pos} to {vip_current}, distance to target: {current_distance}")

        # Check if VIP might be using wall-following (based on position near obstacles)
        if step > 10:
            # Check if VIP is near river/obstacle
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    check_x, check_y = vip_current[0] + dx, vip_current[1] + dy
                    if (0 <= check_x < grid_size and 0 <= check_y < grid_size):
                        if terrain_map[check_x][check_y][0] == 'river':
                            wall_following_used = True
                            break

        obs = next_obs

        if done:
            print(f"Episode completed at step {step + 1}")
            break

    # Analyze results
    print(f"\n✅ VIP Navigation Test Results:")
    print(f"  Total steps observed: {len(vip_positions) - 1}")
    print(f"  Total distance traveled: {total_distance_traveled}")
    print(f"  Average movement per step: {total_distance_traveled / (len(vip_positions) - 1):.2f}")

    # Check final distance to target
    final_pos = vip_positions[-1]
    final_distance = abs(final_pos[0] - vip_target[0]) + abs(final_pos[1] - vip_target[1])
    print(f"  Final distance to target: {final_distance}")

    # Analyze movement patterns
    unique_positions = len(set(vip_positions))
    print(f"  Unique positions visited: {unique_positions} out of {len(vip_positions)} steps")

    if len(stuck_periods) > 0:
        print(f"  ⚠️  Stuck periods detected at steps: {stuck_periods}")
        print(f"  📊 Wall-following should have been used during these periods")
    else:
        print(f"  ✅ No obvious stuck periods detected")

    if wall_following_used:
        print(f"  🎯 VIP navigated near obstacles (potentially using edge-following)")

    # Check if VIP made reasonable progress
    initial_distance = abs(vip_initial[0] - vip_target[0]) + abs(vip_initial[1] - vip_target[1])
    progress_made = initial_distance - final_distance

    if progress_made > 0:
        print(f"  ✅ VIP made progress: {progress_made} units closer to target")
    else:
        print(f"  ⚠️  VIP did not make progress (same or further from target)")

    # Check if VIP got stuck at the end
    if len(vip_positions) >= 4:
        final_positions = vip_positions[-4:]
        final_unique = len(set(final_positions))
        if final_unique <= 2:
            print(f"  ⚠️  VIP appears to be stuck at the end (only {final_unique} positions in last 4 steps)")
        else:
            print(f"  ✅ VIP is still moving actively at the end")

    env.close()

    # Return success if VIP made reasonable progress
    return progress_made > 0 and total_distance_traveled > len(vip_positions) * 0.3


def test_vip_with_different_obstacles():
    """Test VIP navigation with different obstacle configurations"""
    print("\n🧪 Testing VIP Navigation with Different Obstacles...")

    # Test with easy difficulty (smaller grid, fewer obstacles)
    print("\n--- Testing Easy Difficulty ---")
    env_easy = create_dem_ctde_env(
        difficulty="easy",
        global_state_type="concat",
        max_steps=50
    )

    success_easy = test_single_env_scenario(env_easy, "Easy")

    # Test with hard difficulty (more threats, same obstacles)
    print("\n--- Testing Hard Difficulty ---")
    env_hard = create_dem_ctde_env(
        difficulty="hard",
        global_state_type="concat",
        max_steps=80
    )

    success_hard = test_single_env_scenario(env_hard, "Hard")

    return success_easy, success_hard


def test_single_env_scenario(env, difficulty_name):
    """Test a single environment scenario"""
    obs = env.reset()
    initial_info = env.get_global_info()
    vip_initial = initial_info['vip']['pos']
    vip_target = initial_info['vip']['target_pos']

    print(f"VIP {vip_initial} -> Target {vip_target}")

    vip_positions = [vip_initial]

    for step in range(25):
        actions = {agent_id: 0 for agent_id in obs.keys()}  # All stay
        next_obs, rewards, done, info = env.step(actions)

        current_info = env.get_global_info()
        vip_current = current_info['vip']['pos']
        vip_positions.append(vip_current)

        if step % 5 == 0:
            distance = abs(vip_current[0] - vip_target[0]) + abs(vip_current[1] - vip_target[1])
            print(f"  Step {step}: VIP at {vip_current}, distance to target: {distance}")

        obs = next_obs
        if done:
            break

    final_distance = abs(vip_positions[-1][0] - vip_target[0]) + abs(vip_positions[-1][1] - vip_target[1])
    unique_positions = len(set(vip_positions))

    print(f"  Final distance: {final_distance}, Unique positions: {unique_positions}/{len(vip_positions)}")

    env.close()
    return final_distance < 15 and unique_positions > len(vip_positions) * 0.4


if __name__ == "__main__":
    print("=" * 60)
    print("VIP PATHFINDING IMPROVEMENT TESTS")
    print("=" * 60)

    success = True

    # Main obstacle navigation test
    success &= test_vip_obstacle_navigation()

    # Test different difficulty scenarios
    easy_success, hard_success = test_vip_with_different_obstacles()
    success &= easy_success and hard_success

    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED! VIP pathfinding improvements are working well.")
        print("   VIP can now navigate around obstacles more effectively.")
    else:
        print("❌ Some tests failed. VIP pathfinding may need further improvements.")
    print("=" * 60)