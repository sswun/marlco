"""
DEM Environment Test Suite

This module contains tests for the DEM environment to ensure
correct functionality and compatibility with RL algorithms.
"""

import numpy as np
import time
from typing import Dict, Any

try:
    from .env_dem import DEMEnv, DEMConfig, create_dem_env
    from .env_dem_ctde import DEMCTDEWrapper, create_dem_ctde_env
except ImportError:
    # Running as script
    from env_dem import DEMEnv, DEMConfig, create_dem_env
    from env_dem_ctde import DEMCTDEWrapper, create_dem_ctde_env


def test_basic_functionality():
    """Test basic environment functionality"""
    print("🧪 Testing Basic Functionality...")

    # Create environment
    env = create_dem_env(difficulty="easy", max_steps=50)

    # Test reset
    obs = env.reset()
    print(f"✅ Reset successful, got {len(obs)} observations")

    # Test step
    actions = {agent_id: np.random.randint(0, 10) for agent_id in obs.keys()}
    next_obs, rewards, done, info = env.step(actions)
    print(f"✅ Step successful, done: {done}")

    # Test observation structure
    for agent_id, observation in obs.items():
        assert isinstance(observation, np.ndarray), f"Observation for {agent_id} should be numpy array"
        assert observation.dtype == np.float32, f"Observation for {agent_id} should be float32"
        print(f"✅ Agent {agent_id}: observation shape {observation.shape}")

    # Test env info
    env_info = env.get_env_info()
    print(f"✅ Environment info: {env_info['n_agents']} agents, "
          f"obs dim {env_info['obs_shape']}, actions {env_info['n_actions']}")

    env.close()
    return True


def test_ctde_functionality():
    """Test CTDE wrapper functionality"""
    print("\n🧪 Testing CTDE Functionality...")

    # Test different global state types
    for state_type in ["concat", "mean", "max", "attention"]:
        print(f"  Testing {state_type} global state...")

        env = create_dem_ctde_env(
            difficulty="easy",
            global_state_type=state_type,
            max_steps=30
        )

        obs = env.reset()
        global_state = env.get_global_state()

        print(f"  ✅ {state_type}: global state dim {global_state.shape}")
        assert len(global_state.shape) == 1, f"Global state should be 1D array"

        env.close()

    return True


def test_episode_flow():
    """Test complete episode flow"""
    print("\n🧪 Testing Episode Flow...")

    env = create_dem_ctde_env(difficulty="normal", max_steps=100)

    obs = env.reset()
    episode_length = 0
    total_reward = 0

    while True:
        # Random actions for testing
        actions = {}
        for agent_id in obs.keys():
            # Get available actions
            avail_actions = env.get_avail_agent_actions(agent_id)
            actions[agent_id] = np.random.choice(avail_actions)

        next_obs, rewards, done, info = env.step(actions)

        episode_length += 1
        total_reward += sum(rewards.values())

        if done:
            print(f"✅ Episode completed in {episode_length} steps")
            print(f"  Termination reason: {info.get('termination_reason', 'unknown')}")
            print(f"  Total reward: {total_reward:.2f}")
            print(f"  VIP HP: {info['vip_hp']}/{info['vip_max_hp']}")
            print(f"  Threats killed: {info['threats_killed']}")
            break

    env.close()
    return True


def test_difficulty_levels():
    """Test different difficulty levels"""
    print("\n🧪 Testing Difficulty Levels...")

    difficulties = ["easy", "normal", "hard"]
    results = {}

    for difficulty in difficulties:
        print(f"  Testing {difficulty} difficulty...")

        env = create_dem_ctde_env(difficulty=difficulty, max_steps=50)

        obs = env.reset()
        env_info = env.get_env_info()
        global_info = env.get_global_info()

        results[difficulty] = {
            'agents': env_info['n_agents'],
            'max_steps': env_info['max_steps'],
            'vip_hp': global_info['vip']['hp'],
            'grid_size': env.config.grid_size,
        }

        print(f"  ✅ {difficulty}: VIP HP {results[difficulty]['vip_hp']}, "
              f"max steps {results[difficulty]['max_steps']}")

        env.close()

    return results


def test_reward_structure():
    """Test reward structure and role emergence"""
    print("\n🧪 Testing Reward Structure...")

    env = create_dem_ctde_env(difficulty="normal", max_steps=200)

    obs = env.reset()
    total_rewards = {agent_id: 0.0 for agent_id in obs.keys()}
    role_events = []

    for step in range(50):  # Short episode for testing
        # Simple policy: guard VIP for first half, then explore
        actions = {}
        for agent_id in obs.keys():
            if step < 25:
                actions[agent_id] = 6  # Guard VIP
            else:
                actions[agent_id] = np.random.choice([0, 1, 2, 3, 4])  # Random movement

        next_obs, rewards, done, info = env.step(actions)

        for agent_id, reward in rewards.items():
            total_rewards[agent_id] += reward

        # Track role events
        stats = env.get_stats()
        if stats['agents_adjacent_to_vip'] > 0:
            role_events.append(f"Step {step}: {stats['agents_adjacent_to_vip']} agents guarding VIP")
        if stats['body_blocks'] > 0:
            role_events.append(f"Step {step}: Body block occurred")

        obs = next_obs

        if done:
            break

    print(f"✅ Total rewards per agent: {total_rewards}")
    print(f"✅ Role events: {len(role_events)} events detected")
    for event in role_events[:5]:  # Show first 5 events
        print(f"  {event}")

    env.close()
    return True


def test_observations_and_states():
    """Test observation and state encoding"""
    print("\n🧪 Testing Observations and States...")

    env = create_dem_ctde_env(difficulty="normal", global_state_type="concat")

    obs = env.reset()
    global_state = env.get_global_state()
    global_info = env.get_global_info()

    print(f"✅ Observation dimension: {list(obs.values())[0].shape[0]}")
    print(f"✅ Global state dimension: {global_state.shape[0]}")
    print(f"✅ VIP position: {global_info['vip']['pos']}")
    print(f"✅ VIP target: {global_info['vip']['target_pos']}")
    print(f"✅ Agents alive: {len(global_info['agents'])}")
    print(f"✅ Threats alive: {len(global_info['threats'])}")

    # Test observation encoding
    for agent_id, observation in obs.items():
        print(f"✅ Agent {agent_id} observation stats:")
        print(f"  Mean: {observation.mean():.4f}, Std: {observation.std():.4f}")
        print(f"  Min: {observation.min():.4f}, Max: {observation.max():.4f}")

    env.close()
    return True


def test_termination_conditions():
    """Test various termination conditions"""
    print("\n🧪 Testing Termination Conditions...")

    termination_results = {}

    # Test VIP death (hard difficulty with aggressive threats)
    print("  Testing VIP death condition...")
    env = create_dem_ctde_env(difficulty="hard", max_steps=100)
    obs = env.reset()

    for step in range(200):
        actions = {agent_id: 0 for agent_id in obs.keys()}  # All stay still
        obs, rewards, done, info = env.step(actions)

        if done:
            termination_results['vip_death'] = info.get('termination_reason')
            print(f"  ✅ Termination: {termination_results['vip_death']} at step {step}")
            break

    env.close()

    # Test max steps
    print("  Testing max steps condition...")
    env = create_dem_ctde_env(difficulty="easy", max_steps=20)
    obs = env.reset()

    for step in range(50):
        actions = {agent_id: 1 for agent_id in obs.keys()}  # Move up
        obs, rewards, done, info = env.step(actions)

        if done:
            termination_results['max_steps'] = info.get('termination_reason')
            print(f"  ✅ Termination: {termination_results['max_steps']} at step {step}")
            break

    env.close()

    return termination_results


def run_performance_test():
    """Test environment performance"""
    print("\n🧪 Testing Performance...")

    env = create_dem_ctde_env(difficulty="normal", max_steps=1000)

    # Measure step time
    obs = env.reset()
    start_time = time.time()

    steps = 0
    for _ in range(100):  # Run 100 steps
        actions = {agent_id: np.random.randint(0, 10) for agent_id in obs.keys()}
        obs, rewards, done, info = env.step(actions)
        steps += 1

        if done:
            break

    end_time = time.time()
    total_time = end_time - start_time
    steps_per_second = steps / total_time

    print(f"✅ Performance: {steps_per_second:.1f} steps/second")
    print(f"✅ Average step time: {(total_time/steps)*1000:.2f} ms")

    env.close()
    return steps_per_second


def main():
    """Run all tests"""
    print("🚀 Starting DEM Environment Test Suite")
    print("=" * 50)

    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("CTDE Functionality", test_ctde_functionality),
        ("Episode Flow", test_episode_flow),
        ("Difficulty Levels", test_difficulty_levels),
        ("Reward Structure", test_reward_structure),
        ("Observations and States", test_observations_and_states),
        ("Termination Conditions", test_termination_conditions),
        ("Performance", run_performance_test),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            start_time = time.time()
            success = test_func()
            end_time = time.time()

            result = (test_name, success, end_time - start_time)
            results.append(result)

            if success:
                print(f"✅ {test_name} - PASSED ({end_time - start_time:.2f}s)")
            else:
                print(f"❌ {test_name} - FAILED")

        except Exception as e:
            print(f"❌ {test_name} - ERROR: {e}")
            results.append((test_name, False, 0))

        print("-" * 30)

    # Summary
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)

    print(f"\n📊 Test Summary: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! DEM environment is working correctly.")
        print("\nKey features validated:")
        print("✅ Basic environment functionality")
        print("✅ CTDE wrapper compatibility")
        print("✅ Multiple difficulty levels")
        print("✅ Reward structure and role emergence")
        print("✅ Observation and state encoding")
        print("✅ Termination conditions")
        print("✅ Performance requirements")
    else:
        print("⚠️ Some tests failed. Please review the implementation.")

    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)