#!/usr/bin/env python3
"""
Simple consistency test for CM environment core fixes
"""

import sys
import os

# Add project root to path
sys.path.insert(0, '/mnt/j/files/A_projects_extension/code_python_mulitiagent/MARL')

def test_step_format():
    """Test step return format fix"""
    print("Testing step return format fix...")

    try:
        from Env.CM.env_cm import CooperativeMovingEnv
        from Env.CM.config import CMConfig

        # Create environment without rendering
        config = CMConfig(
            grid_size=5,
            n_agents=2,
            max_steps=10,
            render_mode=None
        )
        env = CooperativeMovingEnv(config=config)

        obs, info = env.reset()

        # Test step
        actions = {agent_id: env.get_avail_actions(agent_id)[0] for agent_id in env.agent_ids}
        result = env.step(actions)

        if len(result) == 4:
            obs, rewards, done, info = result
            print("✅ Step returns 4 values (FIXED)")
            print(f"  - observations: {type(obs)} with {len(obs)} agents")
            print(f"  - rewards: {type(rewards)} with {len(rewards)} agents")
            print(f"  - done: {type(done)} = {done}")
            print(f"  - info: {type(info)}")
            return True
        else:
            print(f"❌ Step returns {len(result)} values, expected 4")
            return False

    except Exception as e:
        print(f"❌ Step format test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ctde_format():
    """Test CTDE step return format fix"""
    print("\nTesting CTDE step return format fix...")

    try:
        from Env.CM.env_cm_ctde import CooperativeMovingCTDEEnv
        from Env.CM.config import CMConfig

        # Create CTDE environment
        config = CMConfig(
            grid_size=5,
            n_agents=2,
            max_steps=10,
            render_mode=None
        )
        ctde_env = CooperativeMovingCTDEEnv(
            difficulty="debug",
            global_state_type="concat"
        )

        obs, global_state = ctde_env.reset()

        # Test step
        actions = {agent_id: 0 for agent_id in ctde_env.agent_ids}
        result = ctde_env.step(actions)

        if len(result) == 4:
            obs, rewards, done, info = result
            print("✅ CTDE step returns 4 values (FIXED)")
            print(f"  - global_state in info: {'global_state' in info}")
            return True
        else:
            print(f"❌ CTDE step returns {len(result)} values, expected 4")
            return False

    except Exception as e:
        print(f"❌ CTDE format test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Test basic functionality still works"""
    print("\nTesting basic functionality...")

    try:
        from Env.CM import create_cm_env

        # Create environment
        env = create_cm_env(difficulty="debug", render_mode=None)
        print(f"✅ Environment created with {env.n_agents} agents")

        # Reset
        obs, info = env.reset()
        print(f"✅ Environment reset successful")

        # Get available actions
        for agent_id in env.agent_ids:
            avail_actions = env.get_avail_actions(agent_id)
            print(f"✅ Agent {agent_id} has {len(avail_actions)} available actions")

        # Test a few steps
        for step in range(3):
            actions = {agent_id: env.get_avail_actions(agent_id)[0] for agent_id in env.agent_ids}
            obs, rewards, done, info = env.step(actions)
            print(f"✅ Step {step+1}: done={done}, distance={info['distance_to_goal']:.2f}")

            if done:
                break

        env.close()
        return True

    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_system():
    """Test configuration system"""
    print("\nTesting configuration system...")

    try:
        from Env.CM.config import list_available_configs, get_config_by_name

        configs = list_available_configs()
        print(f"✅ Available configurations: {configs}")

        # Test a few configs
        for config_name in ["easy", "debug", "easy_ctde"]:
            config = get_config_by_name(config_name)
            print(f"✅ Config '{config_name}': {config.n_agents} agents, {config.grid_size}x{config.grid_size}")

        return True

    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("CM Environment Simple Consistency Test")
    print("=" * 50)

    tests = [
        test_step_format,
        test_ctde_format,
        test_basic_functionality,
        test_config_system
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print("\n" + "=" * 50)
    print(f"Simple Consistency Test Results: {passed}/{total} tests passed")
    print("=" * 50)

    if passed == total:
        print("🎉 All simple consistency tests passed!")
        print("✅ Step return format fixed (4 values)")
        print("✅ Basic functionality working")
        print("✅ Configuration system working")
        print("CM environment fixes are complete!")
    else:
        print("⚠️ Some issues remain.")