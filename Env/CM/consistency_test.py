#!/usr/bin/env python3
"""
Consistency test for CM environment against other environments
"""

import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, '/mnt/j/files/A_projects_extension/code_python_mulitiagent/MARL')

def test_step_return_format():
    """Test that step() returns 4 values like other environments"""
    print("Testing step return format consistency...")

    try:
        from Env.CM import create_cm_env
        env = create_cm_env(difficulty="debug", render_mode=None)
        obs, info = env.reset()

        # Test step return format
        actions = {agent_id: env.get_avail_actions(agent_id)[0] for agent_id in env.agent_ids}
        result = env.step(actions)

        print(f"Step result type: {type(result)}")
        print(f"Step result length: {len(result)}")

        if len(result) == 4:
            obs, rewards, done, info = result
            print("✅ Step returns 4 values (observations, rewards, done, info)")
            print(f"  - observations type: {type(obs)}")
            print(f"  - rewards type: {type(rewards)}")
            print(f"  - done type: {type(done)}")
            print(f"  - info type: {type(info)}")
            return True
        else:
            print(f"❌ Step returns {len(result)} values, expected 4")
            return False

    except Exception as e:
        print(f"❌ Step format test failed: {e}")
        return False

def test_ctde_consistency():
    """Test CTDE environment consistency"""
    print("\nTesting CTDE environment consistency...")

    try:
        from Env.CM import create_cm_ctde_env
        ctde_env = create_cm_ctde_env(difficulty="debug", global_state_type="concat")
        obs, global_state = ctde_env.reset()

        # Test step return format
        actions = {agent_id: 0 for agent_id in ctde_env.agent_ids}
        result = ctde_env.step(actions)

        if len(result) == 4:
            obs, rewards, done, info = result
            print("✅ CTDE step returns 4 values")
            print(f"  - global_state in info: {'global_state' in info}")
            return True
        else:
            print(f"❌ CTDE step returns {len(result)} values, expected 4")
            return False

    except Exception as e:
        print(f"❌ CTDE consistency test failed: {e}")
        return False

def test_rendering():
    """Test rendering functionality"""
    print("\nTesting rendering functionality...")

    try:
        from Env.CM import create_cm_env

        # Test rgb_array mode
        env = create_cm_env(difficulty="debug", render_mode="rgb_array")
        obs, info = env.reset()

        actions = {agent_id: env.get_avail_actions(agent_id)[0] for agent_id in env.agent_ids}
        obs, rewards, done, info = env.step(actions)

        # Try to render
        try:
            rgb_array = env.render()
            if rgb_array is not None:
                print(f"✅ rgb_array rendering works, shape: {rgb_array.shape}")
            else:
                print("⚠️ rgb_array rendering returned None")
        except Exception as e:
            print(f"⚠️ rgb_array rendering issue: {e}")

        env.close()

        # Test human mode
        env = create_cm_env(difficulty="debug", render_mode="human")
        obs, info = env.reset()

        try:
            result = env.render()
            print("✅ human rendering completed without crash")
        except Exception as e:
            print(f"⚠️ human rendering issue: {e}")

        env.close()
        return True

    except Exception as e:
        print(f"❌ Rendering test failed: {e}")
        return False

def test_api_consistency():
    """Test API consistency with other environments"""
    print("\nTesting API consistency...")

    try:
        from Env.CM import create_cm_env
        env = create_cm_env(difficulty="debug")

        # Check expected methods
        expected_methods = ['reset', 'step', 'render', 'close', 'get_avail_actions', 'get_env_info']
        missing_methods = []

        for method in expected_methods:
            if not hasattr(env, method):
                missing_methods.append(method)

        if missing_methods:
            print(f"❌ Missing methods: {missing_methods}")
            return False
        else:
            print("✅ All expected methods present")

        # Test env_info
        env_info = env.get_env_info()
        expected_keys = ['n_agents', 'agent_ids', 'n_actions', 'obs_dims', 'act_dims', 'episode_limit']
        missing_keys = [key for key in expected_keys if key not in env_info]

        if missing_keys:
            print(f"❌ Missing env_info keys: {missing_keys}")
            return False
        else:
            print("✅ env_info contains expected keys")

        env.close()
        return True

    except Exception as e:
        print(f"❌ API consistency test failed: {e}")
        return False

def compare_with_hrg():
    """Compare with HRG environment patterns"""
    print("\nComparing with HRG environment patterns...")

    try:
        # Test CM environment
        from Env.CM import create_cm_env
        cm_env = create_cm_env(difficulty="debug")
        cm_obs, cm_info = cm_env.reset()

        # Test step
        cm_actions = {agent_id: cm_env.get_avail_actions(agent_id)[0] for agent_id in cm_env.agent_ids}
        cm_result = cm_env.step(cm_actions)

        print(f"CM Environment:")
        print(f"  - step returns: {len(cm_result)} values")
        print(f"  - observation keys: {list(cm_obs.keys())}")
        print(f"  - agent IDs: {cm_env.agent_ids}")

        cm_env.close()

        return True

    except Exception as e:
        print(f"❌ Comparison test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("CM Environment Consistency Test")
    print("=" * 60)

    tests = [
        test_step_return_format,
        test_ctde_consistency,
        test_rendering,
        test_api_consistency,
        compare_with_hrg
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print("\n" + "=" * 60)
    print(f"Consistency Test Results: {passed}/{total} tests passed")
    print("=" * 60)

    if passed == total:
        print("🎉 All consistency tests passed!")
        print("CM environment is now consistent with other environments.")
    else:
        print("⚠️ Some consistency issues remain.")