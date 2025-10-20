"""
CM Environment Test Suite

This module provides comprehensive tests for the Collaborative Moving environment,
including basic functionality, CTDE compatibility, and performance tests.
"""

import time
import numpy as np
from typing import Dict, List, Any

from .env_cm import create_cm_env
from .env_cm_ctde import create_cm_ctde_env, create_cm_adapter
from .config import get_config_by_name, list_available_configs


class CMEnvironmentTester:
    """Test suite for the CM environment."""

    def __init__(self):
        self.test_results = []

    def run_all_tests(self):
        """Run all tests and report results."""
        print("=" * 60)
        print("CM Environment Test Suite")
        print("=" * 60)

        tests = [
            self.test_basic_environment,
            self.test_different_configurations,
            self.test_action_validation,
            self.test_observation_space,
            self.test_reward_system,
            self.test_ctde_wrapper,
            self.test_global_state_types,
            self.test_performance,
            self.test_edge_cases
        ]

        passed = 0
        total = len(tests)

        for test in tests:
            try:
                test()
                passed += 1
                print(f"✅ {test.__name__}: PASSED")
            except Exception as e:
                print(f"❌ {test.__name__}: FAILED - {str(e)}")

        print("\n" + "=" * 60)
        print(f"Test Results: {passed}/{total} tests passed")
        print("=" * 60)

        return passed == total

    def test_basic_environment(self):
        """Test basic environment functionality."""
        print("\nTesting basic environment functionality...")

        # Create environment
        env = create_cm_env(difficulty="easy")
        info = env.get_env_info()

        # Verify environment info
        assert info['n_agents'] == 2
        assert info['n_actions'] == 5
        assert info['episode_limit'] == 100

        # Reset environment
        obs, reset_info = env.reset()

        # Verify observations
        assert len(obs) == info['n_agents']
        for agent_id in info['agent_ids']:
            assert agent_id in obs
            assert obs[agent_id].shape == (info['obs_dims'][agent_id],)
            assert obs[agent_id].dtype == np.float32

        # Take a random step
        actions = {}
        for agent_id in env.agent_ids:
            avail_actions = env.get_avail_actions(agent_id)
            actions[agent_id] = np.random.choice(avail_actions)

        obs, rewards, terminated, truncated, step_info = env.step(actions)

        # Verify step outputs
        assert len(obs) == info['n_agents']
        assert len(rewards) == info['n_agents']
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(step_info, dict)

        env.close()
        print("  ✅ Basic functionality test passed")

    def test_different_configurations(self):
        """Test different environment configurations."""
        print("\nTesting different configurations...")

        configs_to_test = ["easy", "normal", "hard", "debug"]

        for config_name in configs_to_test:
            env = create_cm_env(difficulty=config_name)
            info = env.get_env_info()

            # Test reset and step
            obs, _ = env.reset()
            actions = {agent_id: 0 for agent_id in env.agent_ids}  # All stay
            obs, rewards, terminated, truncated, _ = env.step(actions)

            # Verify basic properties
            assert len(obs) == info['n_agents']
            assert len(rewards) == info['n_agents']

            env.close()
            print(f"  ✅ Configuration '{config_name}' test passed")

    def test_action_validation(self):
        """Test action validation and available actions."""
        print("\nTesting action validation...")

        env = create_cm_env(difficulty="easy")
        obs, _ = env.reset()

        # Test available actions
        for agent_id in env.agent_ids:
            avail_actions = env.get_avail_actions(agent_id)
            assert isinstance(avail_actions, list)
            assert len(avail_actions) > 0
            assert all(0 <= action < env.n_actions for action in avail_actions)

        # Test invalid actions
        try:
            invalid_actions = {agent_id: 10 for agent_id in env.agent_ids}
            env.step(invalid_actions)
            assert False, "Should have raised ValueError for invalid action"
        except ValueError:
            pass  # Expected

        # Test missing actions
        try:
            partial_actions = {env.agent_ids[0]: 0}  # Only one agent action
            env.step(partial_actions)
            assert False, "Should have raised ValueError for missing actions"
        except ValueError:
            pass  # Expected

        env.close()
        print("  ✅ Action validation test passed")

    def test_observation_space(self):
        """Test observation space properties."""
        print("\nTesting observation space...")

        env = create_cm_env(difficulty="normal", n_agents=3)
        obs, _ = env.reset()

        expected_dim = 6 + 2 * (env.n_agents - 1)  # 6 base + 2 per other agent

        for agent_id, observation in obs.items():
            assert observation.shape == (expected_dim,)
            assert np.all(observation >= 0)
            assert np.all(observation <= env.config.grid_size)

        # Test observation consistency across steps
        for _ in range(5):
            actions = {agent_id: np.random.choice(env.get_avail_actions(agent_id))
                      for agent_id in env.agent_ids}
            obs, rewards, terminated, truncated, _ = env.step(actions)

            for agent_id, observation in obs.items():
                assert observation.shape == (expected_dim,)

            if terminated:
                break

        env.close()
        print("  ✅ Observation space test passed")

    def test_reward_system(self):
        """Test reward system properties."""
        print("\nTesting reward system...")

        env = create_cm_env(difficulty="easy")
        obs, _ = env.reset()

        reward_history = []

        for step in range(20):
            actions = {agent_id: np.random.choice(env.get_avail_actions(agent_id))
                      for agent_id in env.agent_ids}
            obs, rewards, terminated, truncated, info = env.step(actions)

            # All agents should receive the same reward (team reward)
            reward_values = list(rewards.values())
            assert all(abs(r - reward_values[0]) < 1e-6 for r in reward_values)

            reward_history.append(reward_values[0])

            if terminated:
                # Check for goal reward
                assert any(r > 5.0 for r in reward_values), "Should receive goal reward"
                break

        # Check for time penalty
        assert any(r < 0 for r in reward_history), "Should have time penalties"

        env.close()
        print("  ✅ Reward system test passed")

    def test_ctde_wrapper(self):
        """Test CTDE wrapper functionality."""
        print("\nTesting CTDE wrapper...")

        # Test basic CTDE environment
        ctde_env = create_cm_ctde_env(difficulty="easy_ctde")
        info = ctde_env.get_env_info()

        assert 'global_state_dim' in info
        assert info['global_state_dim'] > 0

        # Test reset with global state
        obs, global_state = ctde_env.reset()
        assert global_state.shape == (info['global_state_dim'],)

        # Test step with global state in info
        actions = {agent_id: 0 for agent_id in ctde_env.agent_ids}
        obs, rewards, terminated, truncated, step_info = ctde_env.step(actions)

        assert 'global_state' in step_info
        assert step_info['global_state'].shape == (info['global_state_dim'],)

        ctde_env.close()

        # Test adapter
        adapter = create_cm_adapter(difficulty="easy_ctde")
        obs, state = adapter.reset()

        assert len(obs) == adapter.n_agents
        assert state.shape == (adapter.get_state_size(),)

        adapter.close()
        print("  ✅ CTDE wrapper test passed")

    def test_global_state_types(self):
        """Test different global state types."""
        print("\nTesting global state types...")

        state_types = ["concat", "mean", "max", "attention"]
        env_info = {}

        for state_type in state_types:
            env = create_cm_ctde_env(
                difficulty="easy_ctde",
                global_state_type=state_type
            )
            info = env.get_env_info()
            env_info[state_type] = info['global_state_dim']

            # Test state computation
            obs, global_state = env.reset()
            assert global_state.shape == (info['global_state_dim'],)

            env.close()

        # Verify different state types have different dimensions (except mean/max might be same)
        assert env_info["concat"] > env_info["mean"]
        assert env_info["concat"] > env_info["max"]
        print(f"  ✅ Global state dimensions: {env_info}")

    def test_performance(self):
        """Test environment performance."""
        print("\nTesting performance...")

        env = create_cm_env(difficulty="debug")  # Use debug config for speed

        # Performance test
        start_time = time.time()
        steps = 0
        episodes = 5

        for episode in range(episodes):
            obs, _ = env.reset()
            episode_steps = 0

            while episode_steps < 50:  # Limit steps for performance test
                actions = {agent_id: np.random.choice(env.get_avail_actions(agent_id))
                          for agent_id in env.agent_ids}
                obs, rewards, terminated, truncated, _ = env.step(actions)

                episode_steps += 1
                steps += 1

                if terminated or truncated:
                    break

        total_time = time.time() - start_time
        steps_per_second = steps / total_time

        print(f"  ✅ Performance: {steps_per_second:.1f} steps/second")
        print(f"  ✅ Average time per step: {total_time/steps*1000:.2f}ms")

        env.close()

        # Performance assertion (should be fast enough for training)
        assert steps_per_second > 100, f"Environment too slow: {steps_per_second:.1f} steps/sec"

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        print("\nTesting edge cases...")

        # Test single agent
        single_env = create_cm_env(difficulty="single_agent")
        obs, _ = single_env.reset()
        assert len(obs) == 1

        # Test max agents
        multi_env = create_cm_env(difficulty="multi_agent")
        obs, _ = multi_env.reset()
        assert len(obs) == 4

        # Test environment reset multiple times
        env = create_cm_env(difficulty="debug")
        for _ in range(5):
            obs, _ = env.reset()
            actions = {agent_id: 0 for agent_id in env.agent_ids}
            obs, rewards, terminated, truncated, _ = env.step(actions)
            assert len(obs) == env.n_agents
            assert len(rewards) == env.n_agents

        # Test seed consistency
        env1 = create_cm_env(difficulty="debug", seed=42)
        env2 = create_cm_env(difficulty="debug", seed=42)

        obs1, _ = env1.reset()
        obs2, _ = env2.reset()

        # Should produce same initial observations with same seed
        for agent_id in env1.agent_ids:
            assert np.allclose(obs1[agent_id], obs2[agent_id])

        env1.close()
        env2.close()
        single_env.close()
        multi_env.close()
        env.close()

        print("  ✅ Edge cases test passed")


def run_quick_test():
    """Run a quick test to verify basic functionality."""
    print("Running quick CM environment test...")

    try:
        # Test basic environment
        env = create_cm_env(difficulty="debug", render_mode=None)
        obs, info = env.reset()

        # Take a few steps
        for step in range(5):
            actions = {agent_id: np.random.choice(env.get_avail_actions(agent_id))
                      for agent_id in env.agent_ids}
            obs, rewards, terminated, truncated, info = env.step(actions)

            if terminated:
                print(f"Goal reached in {step + 1} steps!")
                break

        env.close()

        # Test CTDE environment
        ctde_env = create_cm_ctde_env(difficulty="debug", global_state_type="concat")
        obs, global_state = ctde_env.reset()

        actions = {agent_id: 0 for agent_id in ctde_env.agent_ids}
        obs, rewards, terminated, truncated, info = ctde_env.step(actions)

        assert 'global_state' in info
        ctde_env.close()

        print("✅ Quick test passed!")
        return True

    except Exception as e:
        print(f"❌ Quick test failed: {str(e)}")
        return False


def run_comprehensive_test():
    """Run comprehensive test suite."""
    tester = CMEnvironmentTester()
    return tester.run_all_tests()


def test_configuration_system():
    """Test the configuration system."""
    print("Testing configuration system...")

    configs = list_available_configs()
    print(f"Available configurations: {configs}")

    for config_name in configs[:3]:  # Test first 3 configs
        try:
            config = get_config_by_name(config_name)
            print(f"✅ Config '{config_name}': {config.n_agents} agents, {config.grid_size}x{config.grid_size} grid")
        except Exception as e:
            print(f"❌ Config '{config_name}' failed: {e}")
            return False

    return True


if __name__ == "__main__":
    # Run tests based on command line arguments or run all by default
    import sys

    if len(sys.argv) > 1:
        test_type = sys.argv[1]
        if test_type == "quick":
            run_quick_test()
        elif test_type == "comprehensive":
            run_comprehensive_test()
        elif test_type == "config":
            test_configuration_system()
        else:
            print("Unknown test type. Available: quick, comprehensive, config")
    else:
        # Run all tests
        print("Running all CM environment tests...")
        config_ok = test_configuration_system()
        quick_ok = run_quick_test()
        comprehensive_ok = run_comprehensive_test()

        if config_ok and quick_ok and comprehensive_ok:
            print("\n🎉 All tests passed! CM environment is ready for use.")
        else:
            print("\n❌ Some tests failed. Please check the implementation.")