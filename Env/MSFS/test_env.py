#!/usr/bin/env python3
"""
Test script for MSFS environment functionality
"""

import sys
import os
import numpy as np
import time

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_basic_environment():
    """Test basic MSFS environment functionality"""
    print("🧪 Testing Basic MSFS Environment...")

    try:
        from Env.MSFS.env_msfs import create_msfs_env
        print("✅ Import successful")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

    # Test environment creation
    print("\n📋 Creating environment...")
    env = create_msfs_env(difficulty="easy", max_steps=20)
    print(f"✅ Environment created")

    # Test reset
    print("\n📋 Testing reset...")
    obs = env.reset()
    print(f"✅ Environment reset with {len(obs)} agents")
    print(f"   Observation shape: {list(obs.values())[0].shape}")

    # Test step
    print("\n📋 Testing step...")
    actions = {agent_id: np.random.randint(0, 8) for agent_id in obs.keys()}
    obs, rewards, done, info = env.step(actions)
    print(f"✅ Step executed")
    print(f"   Rewards: {rewards}")
    print(f"   Done: {done}")
    print(f"   Info keys: {list(info.keys())}")

    # Test full episode
    print("\n📋 Testing full episode...")
    obs = env.reset()
    total_reward = 0
    steps = 0

    for step in range(20):
        actions = {agent_id: np.random.randint(0, 8) for agent_id in obs.keys()}
        obs, rewards, done, info = env.step(actions)
        total_reward += sum(rewards.values())
        steps += 1

        if done:
            break

    print(f"✅ Episode completed in {steps} steps")
    print(f"   Total reward: {total_reward:.2f}")
    print(f"   Orders completed: {info.get('orders_completed', 0)}")

    env.close()
    print("✅ Basic environment test passed!")
    return True

def test_ctde_environment():
    """Test CTDE environment functionality"""
    print("\n🧪 Testing CTDE Environment...")

    try:
        from Env.MSFS.env_msfs_ctde import create_msfs_ctde_env
        print("✅ Import successful")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

    # Test CTDE environment creation
    print("\n📋 Creating CTDE environment...")
    ctde_env = create_msfs_ctde_env(difficulty="normal", max_steps=20, global_state_type="concat")
    print(f"✅ CTDE environment created")

    # Test environment info
    env_info = ctde_env.get_env_info()
    print(f"   Agents: {env_info['n_agents']}")
    print(f"   Global state dim: {env_info['global_state_dim']}")
    print(f"   Observation shape: {env_info['obs_shape']}")
    print(f"   Actions: {env_info['n_actions']}")

    # Test global state
    print("\n📋 Testing global state...")
    obs = ctde_env.reset()
    global_state = ctde_env.get_global_state()
    print(f"✅ Global state shape: {global_state.shape}")

    # Test global info
    global_info = ctde_env.get_global_info()
    print(f"✅ Global info keys: {list(global_info.keys())}")
    print(f"   Stats: {global_info['stats']}")

    ctde_env.close()
    print("✅ CTDE environment test passed!")
    return True

def test_configurations():
    """Test different environment configurations"""
    print("\n🧪 Testing Different Configurations...")

    try:
        from Env.MSFS.env_msfs_ctde import create_msfs_ctde_env
        print("✅ Import successful")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

    configs = [
        ("easy", {"difficulty": "easy"}),
        ("normal", {"difficulty": "normal"}),
        ("hard", {"difficulty": "hard"}),
        ("single_agent", {"difficulty": "easy", "num_agents": 1}),
        ("three_agent", {"difficulty": "normal", "num_agents": 3}),
    ]

    for config_name, config_params in configs:
        print(f"\n📋 Testing {config_name} configuration...")
        try:
            env = create_msfs_ctde_env(**config_params, max_steps=10)
            obs = env.reset()

            # Run a few steps
            for step in range(5):
                actions = {agent_id: np.random.randint(0, 8) for agent_id in obs.keys()}
                obs, rewards, done, info = env.step(actions)

                if done:
                    break

            env.close()
            print(f"✅ {config_name} configuration passed")

        except Exception as e:
            print(f"❌ {config_name} configuration failed: {e}")
            return False

    print("✅ All configuration tests passed!")
    return True

def test_rendering():
    """Test rendering functionality"""
    print("\n🧪 Testing Rendering...")

    try:
        from Env.MSFS.env_msfs_ctde import create_msfs_ctde_env
        print("✅ Import successful")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

    # Test RGB array rendering
    print("\n📋 Testing RGB array rendering...")
    env = create_msfs_ctde_env(
        difficulty="easy",
        render_mode="rgb_array",
        max_steps=10
    )

    obs = env.reset()
    rgb_array = env.render(mode="rgb_array")

    if rgb_array is not None:
        print(f"✅ RGB array rendering successful")
        print(f"   Array shape: {rgb_array.shape}")
        print(f"   Data type: {rgb_array.dtype}")
    else:
        print("❌ RGB array rendering failed")
        env.close()
        return False

    # Test a few steps with rendering
    for step in range(3):
        actions = {agent_id: np.random.randint(0, 8) for agent_id in obs.keys()}
        obs, rewards, done, info = env.step(actions)

        rgb_array = env.render(mode="rgb_array")
        if rgb_array is None:
            print(f"❌ Step {step + 1} rendering failed")
            env.close()
            return False

        if done:
            break

    env.close()
    print("✅ Rendering test passed!")
    return True

def main():
    """Run all tests"""
    print("🚀 Starting MSFS Environment Tests")
    print("=" * 50)

    tests = [
        ("Basic Environment", test_basic_environment),
        ("CTDE Environment", test_ctde_environment),
        ("Configurations", test_configurations),
        ("Rendering", test_rendering),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            print(f"\n🧪 Running {test_name}...")
            success = test_func()
            results.append((test_name, success))

            if success:
                print(f"✅ {test_name} - PASSED")
            else:
                print(f"❌ {test_name} - FAILED")

        except Exception as e:
            print(f"❌ {test_name} - ERROR: {e}")
            results.append((test_name, False))

        print("-" * 30)

    # Summary
    passed = sum(1 for _, success in results if success)
    total = len(results)

    print(f"\n📊 Test Summary: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! MSFS environment is working correctly.")
        print("\n📝 Usage Examples:")
        print("  # Basic environment:")
        print("  from Env.MSFS.env_msfs import create_msfs_env")
        print("  env = create_msfs_env(difficulty='normal')")
        print("  obs = env.reset()")
        print("  actions = {agent_id: env.action_space.sample() for agent_id in obs}")
        print("  obs, rewards, done, info = env.step(actions)")
        print()
        print("  # CTDE environment:")
        print("  from Env.MSFS.env_msfs_ctde import create_msfs_ctde_env")
        print("  env = create_msfs_ctde_env(difficulty='normal', global_state_type='concat')")
        print("  obs = env.reset()")
        print("  global_state = env.get_global_state()")
        print()
        print("  # With rendering:")
        print("  env = create_msfs_ctde_env(difficulty='normal', render_mode='human')")
        print("  env.render()  # Opens visualization window")
        print("  env.close()   # Don't forget to close!")
    else:
        print("⚠️ Some tests failed. Please check the implementation.")

    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)