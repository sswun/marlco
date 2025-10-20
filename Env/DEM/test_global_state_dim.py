#!/usr/bin/env python3
"""
Test script to verify DEM global state dimensions
"""

import numpy as np
from Env.DEM.env_dem_ctde import create_dem_ctde_env


def test_global_state_dimensions():
    """Test global state dimensions calculation"""
    print("🧪 Testing Global State Dimensions...")

    # Create environment
    env = create_dem_ctde_env(difficulty="normal", global_state_type="concat")

    # Reset environment
    obs = env.reset()
    global_state = env.get_global_state()
    global_info = env.get_global_info()

    print(f"Actual global state shape: {global_state.shape}")
    print(f"Expected dimensions from calculation: {env.global_state_dim}")

    # Manual calculation based on actual code
    vip_dim = 4  # pos.x, pos.y, hp_ratio, is_under_attack

    config = env.config
    agents_dim = config.num_agents * 4  # pos.x, pos.y, hp_ratio, is_guarding
    threats_dim = 5 * 4  # pos.x, pos.y, hp_ratio, is_rusher
    stats_dim = 5  # vip_distance, agents_adjacent, agents_ahead, agent_spread, step_ratio

    calculated_total = vip_dim + agents_dim + threats_dim + stats_dim
    print(f"\nManual dimension calculation:")
    print(f"  VIP: {vip_dim}")
    print(f"  Agents ({config.num_agents} × 4): {agents_dim}")
    print(f"  Threats (5 × 4): {threats_dim}")
    print(f"  Statistics: {stats_dim}")
    print(f"  Total: {calculated_total}")

    # Check if dimensions match
    actual_dim = global_state.shape[0]
    expected_dim = env.global_state_dim

    if actual_dim == expected_dim:
        print(f"\n✅ Dimensions match: {actual_dim}")
    else:
        print(f"\n❌ Dimension mismatch!")
        print(f"  Actual: {actual_dim}")
        print(f"  Expected: {expected_dim}")
        print(f"  Manual calc: {calculated_total}")

    # Print actual global state content for debugging
    print(f"\nGlobal state content (first 20 values):")
    print(f"  {global_state[:20]}")

    if len(global_state) > 20:
        print(f"  ... (remaining {len(global_state) - 20} values)")

    env.close()
    return actual_dim == expected_dim


def test_difficulty_variations():
    """Test global state dimensions with different difficulty settings"""
    print("\n🧪 Testing Difficulty Variations...")

    difficulties = ["easy", "normal", "hard"]

    for difficulty in difficulties:
        print(f"\nTesting {difficulty} difficulty:")

        env = create_dem_ctde_env(difficulty=difficulty, global_state_type="concat")
        obs = env.reset()
        global_state = env.get_global_state()

        print(f"  Config num_agents: {env.config.num_agents}")
        print(f"  Global state shape: {global_state.shape}")
        print(f"  Expected: {env.global_state_dim}")

        # Check if match
        actual = global_state.shape[0]
        expected = env.global_state_dim
        status = "✅" if actual == expected else "❌"
        print(f"  Status: {status}")

        env.close()


def analyze_global_state_components():
    """Analyze each component of the global state"""
    print("\n🧪 Analyzing Global State Components...")

    env = create_dem_ctde_env(difficulty="normal", global_state_type="concat")
    obs = env.reset()
    global_state = env.get_global_state()
    global_info = env.get_global_info()

    print("Global state breakdown:")

    # VIP component (4 dimensions)
    idx = 0
    vip_info = global_info['vip']
    print(f"VIP (4):")
    print(f"  pos.x: {global_state[idx]:.3f} (actual: {vip_info['pos'][0]}/{env.config.grid_size} = {vip_info['pos'][0]/env.config.grid_size:.3f})")
    idx += 1
    print(f"  pos.y: {global_state[idx]:.3f} (actual: {vip_info['pos'][1]}/{env.config.grid_size} = {vip_info['pos'][1]/env.config.grid_size:.3f})")
    idx += 1
    print(f"  hp_ratio: {global_state[idx]:.3f} (actual: {vip_info['hp']}/{vip_info.get('max_hp', 60)} = {vip_info['hp']/vip_info.get('max_hp', 60):.3f})")
    idx += 1
    print(f"  is_under_attack: {global_state[idx]:.3f}")
    idx += 1

    # Agents component (num_agents × 4 dimensions)
    print(f"\nAgents ({env.config.num_agents} × 4 = {env.config.num_agents * 4}):")
    agents_info = global_info['agents']
    for i, (agent_id, agent_pos) in enumerate(agents_info.items()):
        print(f"  Agent {i+1} ({agent_id}):")
        print(f"    pos.x: {global_state[idx]:.3f}")
        idx += 1
        print(f"    pos.y: {global_state[idx]:.3f}")
        idx += 1
        print(f"    hp_ratio: {global_state[idx]:.3f}")
        idx += 1
        print(f"    is_guarding: {global_state[idx]:.3f}")
        idx += 1

    # Threats component (5 × 4 = 20 dimensions)
    print(f"\nThreats (5 × 4 = 20):")
    threats_info = global_info['threats']
    for i in range(5):
        print(f"  Threat slot {i+1}:")
        if i < len(threats_info):
            tx, ty, ttype = threats_info[i]
            print(f"    type: {global_state[idx]:.3f} ({'rusher' if ttype == 'rusher' else 'shooter'})")
            idx += 1
            print(f"    pos.x: {global_state[idx]:.3f} ({tx}/{env.config.grid_size} = {tx/env.config.grid_size:.3f})")
            idx += 1
            print(f"    pos.y: {global_state[idx]:.3f} ({ty}/{env.config.grid_size} = {ty/env.config.grid_size:.3f})")
            idx += 1
            print(f"    hp_ratio: {global_state[idx]:.3f}")
            idx += 1
        else:
            print(f"    (empty threat): {global_state[idx:idx+4]}")
            idx += 4

    # Statistics component (5 dimensions)
    print(f"\nStatistics (5):")
    stats_info = global_info['stats']
    print(f"  vip_distance_to_target: {global_state[idx]:.3f}")
    idx += 1
    print(f"  agents_adjacent_to_vip: {global_state[idx]:.3f}")
    idx += 1
    print(f"  agents_ahead_of_vip: {global_state[idx]:.3f}")
    idx += 1
    print(f"  agent_spread: {global_state[idx]:.3f}")
    idx += 1
    print(f"  step_ratio: {global_state[idx]:.3f}")
    idx += 1

    print(f"\nTotal indices used: {idx}")
    print(f"Actual global state length: {len(global_state)}")

    env.close()


if __name__ == "__main__":
    print("=" * 60)
    print("DEM GLOBAL STATE DIMENSIONS ANALYSIS")
    print("=" * 60)

    success = True

    success &= test_global_state_dimensions()
    test_difficulty_variations()
    analyze_global_state_components()

    print("\n" + "=" * 60)
    if success:
        print("🎉 Global state dimensions are correct!")
    else:
        print("❌ Global state dimension issues detected!")
    print("=" * 60)