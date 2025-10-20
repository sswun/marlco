#!/usr/bin/env python3
"""
Quick test script for CM environment
"""

import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

def test_cm_environment():
    """Test CM environment functionality"""
    print("Testing CM Environment...")

    try:
        from Env.CM import create_cm_env, create_cm_ctde_env
        print("✅ Import successful")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

    # Test basic environment
    try:
        env = create_cm_env(difficulty="debug", render_mode=None)
        print(f"✅ Basic environment created with {env.n_agents} agents")

        # Reset environment
        obs, info = env.reset()
        print(f"✅ Environment reset successful")
        print(f"   Observation shape: {list(obs.values())[0].shape}")

        # Test one step
        actions = {}
        for agent_id in env.agent_ids:
            avail_actions = env.get_avail_actions(agent_id)
            actions[agent_id] = avail_actions[0]

        obs, rewards, terminated, truncated, step_info = env.step(actions)
        print(f"✅ Step executed successfully")
        print(f"   Rewards: {list(rewards.values())}")
        print(f"   Distance to goal: {step_info['distance_to_goal']:.2f}")

        env.close()

    except Exception as e:
        print(f"❌ Basic environment test failed: {e}")
        return False

    # Test CTDE environment
    try:
        ctde_env = create_cm_ctde_env(difficulty="debug", global_state_type="concat")
        print(f"✅ CTDE environment created")

        obs, global_state = ctde_env.reset()
        print(f"✅ CTDE environment reset successful")
        print(f"   Global state shape: {global_state.shape}")

        # Test CTDE step
        actions = {agent_id: 0 for agent_id in ctde_env.agent_ids}
        obs, rewards, terminated, truncated, info = ctde_env.step(actions)
        print(f"✅ CTDE step executed successfully")
        print(f"   Global state in info: {info['global_state'].shape}")

        ctde_env.close()

    except Exception as e:
        print(f"❌ CTDE environment test failed: {e}")
        return False

    # Test configuration system
    try:
        from Env.CM.config import list_available_configs, get_config_by_name
        configs = list_available_configs()
        print(f"✅ Configuration system works")
        print(f"   Available configs: {configs}")

        config = get_config_by_name("easy")
        print(f"✅ Easy config loaded: {config.n_agents} agents, {config.grid_size}x{config.grid_size} grid")

    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

    # Test random episode
    try:
        env = create_cm_env(difficulty="debug", render_mode=None)
        obs, info = env.reset()

        total_reward = 0
        steps = 0
        max_steps = 20

        while steps < max_steps:
            actions = {}
            for agent_id in env.agent_ids:
                avail_actions = env.get_avail_actions(agent_id)
                actions[agent_id] = np.random.choice(avail_actions)

            obs, rewards, terminated, truncated, step_info = env.step(actions)

            total_reward += list(rewards.values())[0]
            steps += 1

            if terminated:
                print(f"✅ Goal reached in {steps} steps!")
                break
            elif truncated:
                print(f"⚠️ Episode truncated after {steps} steps")
                break

        print(f"✅ Random episode completed: {steps} steps, total reward: {total_reward:.3f}")
        env.close()

    except Exception as e:
        print(f"❌ Random episode test failed: {e}")
        return False

    return True


def test_performance():
    """Test environment performance"""
    print("\nTesting performance...")

    try:
        from Env.CM import create_cm_env
        import time

        env = create_cm_env(difficulty="debug", render_mode=None)
        obs, info = env.reset()

        start_time = time.time()
        num_steps = 100

        for _ in range(num_steps):
            actions = {}
            for agent_id in env.agent_ids:
                avail_actions = env.get_avail_actions(agent_id)
                actions[agent_id] = np.random.choice(avail_actions)

            obs, rewards, terminated, truncated, info = env.step(actions)

            if terminated or truncated:
                obs, info = env.reset()

        total_time = time.time() - start_time
        steps_per_second = num_steps / total_time

        print(f"✅ Performance test: {steps_per_second:.1f} steps/second")

        if steps_per_second > 100:
            print("   🚀 Excellent performance")
        elif steps_per_second > 50:
            print("   ✅ Good performance")
        else:
            print("   ⚠️ Performance could be better")

        env.close()
        return True

    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("CM Environment Quick Test")
    print("=" * 50)

    success = test_cm_environment()

    if success:
        success = test_performance()

    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed! CM environment is ready.")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    print("=" * 50)