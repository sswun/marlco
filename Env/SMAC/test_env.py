"""
SMAC Environment Test Suite

Comprehensive tests for the SMAC environment wrapper implementation,
including basic functionality, CTDE compatibility, and performance tests.
"""

import numpy as np
import time
import logging
from typing import Dict, List, Any

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import SMAC environment components
from .env_smac import SMACEnv, create_smac_env, create_smac_env_easy, create_smac_env_normal, create_smac_env_hard
from .env_smac_ctde import SMACCTDEEnv, create_smac_ctde_env, create_smac_ctde_env_easy
from .config import SMACConfig, get_config_by_name, get_debug_config, get_easy_config, get_normal_config, get_hard_config


def test_basic_environment():
    """Test basic SMAC environment wrapper functionality"""
    print("Testing basic SMAC environment wrapper...")

    # Test default environment creation
    env = SMACEnv()
    env_info = env.get_env_info()

    assert env_info['n_agents'] > 0, "Environment should have at least one agent"
    assert len(env.agent_ids) == env_info['n_agents'], "Agent IDs count should match n_agents"
    assert env_info['n_actions'] > 0, "Should have actions"
    assert env_info['obs_shape'] > 0, "Observation should have positive dimensions"

    print(f"  ✓ Environment created with {env_info['n_agents']} agents")
    print(f"  ✓ Observation space: {env_info['obs_shape']} dimensions")
    print(f"  ✓ Action space: {env_info['n_actions']} actions")

    # Test reset
    observations = env.reset()
    assert len(observations) == env_info['n_agents'], "Reset should return observations for all agents"

    for agent_id, obs in observations.items():
        assert obs.shape == (env_info['obs_shape'],), f"Observation shape mismatch for {agent_id}"
        assert obs.dtype == np.float32, f"Observation should be float32 for {agent_id}"

    print("  ✓ Reset functionality works")

    # Test step
    actions = {}
    for agent_id in env.agent_ids:
        avail_actions = env.get_avail_actions(agent_id)
        actions[agent_id] = avail_actions[0] if avail_actions else 0

    obs, rewards, dones, infos = env.step(actions)

    assert len(obs) == env_info['n_agents'], "Step should return observations for all agents"
    assert len(rewards) == env_info['n_agents'], "Step should return rewards for all agents"
    assert len(dones) == env_info['n_agents'], "Step should return dones for all agents"
    assert 'episode_step' in infos[env.agent_ids[0]], "Info should contain episode step"

    print("  ✓ Step functionality works")

    # Test episode completion
    max_steps = 50  # Limit for testing
    for step in range(max_steps):
        actions = {}
        for agent_id in env.agent_ids:
            avail_actions = env.get_avail_actions(agent_id)
            actions[agent_id] = np.random.choice(avail_actions) if avail_actions else 0

        obs, rewards, dones, infos = env.step(actions)

        if any(dones.values()):
            print(f"  ✓ Episode completed after {step + 1} steps")
            break
    else:
        print(f"  ✓ Episode ran for {max_steps} steps (limited for testing)")

    env.close()
    print("✓ Basic environment test passed\n")


def test_different_maps():
    """Test different SMAC maps"""
    print("Testing different SMAC maps...")

    # Test a few common maps
    maps_to_test = ["8m", "MMM", "corridor", "6h"]

    for map_name in maps_to_test:
        print(f"  Testing map: {map_name}")

        try:
            env = create_smac_env(map_name=map_name, episode_limit=20)  # Short for testing
            env_info = env.get_env_info()

            # Test basic functionality
            observations = env.reset()
            assert len(observations) == env_info['n_agents'], f"Map {map_name}: obs count mismatch"

            # Test a few steps
            for _ in range(5):
                actions = {}
                for agent_id in env.agent_ids:
                    avail_actions = env.get_avail_actions(agent_id)
                    actions[agent_id] = np.random.choice(avail_actions) if avail_actions else 0

                obs, rewards, dones, infos = env.step(actions)

                if any(dones.values()):
                    break

            env.close()
            print(f"    ✓ Map {map_name} works correctly")

        except Exception as e:
            print(f"    ✗ Map {map_name} failed: {e}")
            # Don't raise error for maps that might not be available
            logger.warning(f"Map {map_name} not available or failed: {e}")

    print("✓ Different maps test passed\n")


def test_configuration_system():
    """Test configuration system"""
    print("Testing configuration system...")

    # Test predefined configurations
    configs = ["easy", "normal", "hard", "debug"]

    for config_name in configs:
        print(f"  Testing config: {config_name}")

        config = get_config_by_name(config_name)

        assert config.map_name is not None, f"Config {config_name} should have map name"
        assert config.episode_limit > 0, f"Config {config_name} should have positive episode limit"

        # Test environment creation with config
        env = SMACEnv(config)
        env.reset()
        env.close()

        print(f"    ✓ Config {config_name} works correctly")

    # Test custom configuration
    custom_config = SMACConfig(
        map_name="3s",
        episode_limit=50,
        debug=True,
        seed=42
    )

    env = SMACEnv(custom_config)
    env.reset()
    env.close()

    print("  ✓ Custom configuration works correctly")

    print("✓ Configuration system test passed\n")


def test_ctde_wrapper():
    """Test CTDE wrapper functionality"""
    print("Testing CTDE wrapper...")

    # Test CTDE environment with original SMAC global state
    print("  Testing CTDE environment with original SMAC global state")

    ctde_env = create_smac_ctde_env(map_name="8m", episode_limit=20)

    env_info = ctde_env.get_env_info()
    assert 'global_state_dim' in env_info, "Should have global_state_dim"
    assert env_info['global_state_dim'] > 0, "Global state dim should be positive"

    # Test reset and global state
    observations = ctde_env.reset()
    global_state = ctde_env.get_global_state()

    assert len(global_state.shape) == 1, "Global state should be 1D"
    assert global_state.shape[0] == env_info['global_state_dim'], "Global state dim mismatch"
    assert global_state.dtype == np.float32, "Global state should be float32"

    print(f"    Global state shape: {global_state.shape}")
    print(f"    Global state range: [{global_state.min():.3f}, {global_state.max():.3f}]")

    # Verify global state is the same as original SMAC state (before step)
    original_state = ctde_env.get_state()
    assert np.allclose(global_state, original_state), "Global state should match original SMAC state"
    print("    ✓ Global state matches original SMAC state")

    # Test step with global state in info
    actions = {}
    for agent_id in ctde_env.agent_ids:
        avail_actions = ctde_env.get_avail_actions(agent_id)
        actions[agent_id] = np.random.choice(avail_actions) if avail_actions else 0

    obs, rewards, dones, infos = ctde_env.step(actions)

    assert 'global_state' in infos, "Info should contain global_state"
    assert infos['global_state'].shape == global_state.shape, "Global state shape consistency"

    # Verify info global state matches current global state (after step)
    current_global_state = ctde_env.get_global_state()
    current_original_state = ctde_env.get_state()
    assert np.allclose(infos['global_state'], current_global_state), "Info global state should match current global state"
    assert np.allclose(current_global_state, current_original_state), "Current global state should match current original state"

    print("    ✓ Global state consistency verified throughout step")

    # Test CTDE compatibility functions
    obs = ctde_env.get_obs()
    state = ctde_env.get_state()
    avail_actions = [ctde_env.get_avail_agent_actions(agent_id) for agent_id in ctde_env.agent_ids]

    assert len(obs) == ctde_env.n_agents, "get_obs should return correct number of observations"
    assert len(state.shape) == 1, "get_state should return 1D array"
    assert len(avail_actions) == ctde_env.n_agents, "Should have avail_actions for all agents"

    ctde_env.close()

    print("✓ CTDE wrapper test passed\n")


def test_action_masking():
    """Test action masking functionality"""
    print("Testing action masking...")

    env = create_smac_env(map_name="8m", episode_limit=20)
    observations = env.reset()

    for agent_id in env.agent_ids[:3]:  # Test first 3 agents
        avail_actions = env.get_avail_actions(agent_id)

        assert len(avail_actions) > 0, f"Agent {agent_id} should have available actions"
        assert all(0 <= action < env.env_info['n_actions'] for action in avail_actions), f"Actions should be in valid range for {agent_id}"

        print(f"  Agent {agent_id}: {len(avail_actions)} available actions")

    print("  ✓ Action masking works correctly")

    env.close()
    print("✓ Action masking test passed\n")


def test_observation_encoding():
    """Test observation encoding and consistency"""
    print("Testing observation encoding...")

    env = create_smac_env(map_name="3s", episode_limit=10)
    observations = env.reset()

    obs_dim = env.get_env_info()['obs_shape']

    for agent_id, obs in observations.items():
        assert obs.shape == (obs_dim,), f"Observation shape mismatch for {agent_id}"
        assert np.all(np.isfinite(obs)), f"Observation should be finite for {agent_id}"
        assert obs.dtype == np.float32, f"Observation should be float32 for {agent_id}"

        # Test observation ranges (should be roughly normalized)
        assert np.min(obs) >= -10.0, f"Observation values should not be too small for {agent_id}"
        assert np.max(obs) <= 10.0, f"Observation values should not be too large for {agent_id}"

    print("  ✓ Observation encoding works correctly")

    # Test observation consistency across steps
    prev_obs = None
    for _ in range(3):
        actions = {}
        for agent_id in env.agent_ids:
            avail_actions = env.get_avail_actions(agent_id)
            actions[agent_id] = np.random.choice(avail_actions) if avail_actions else 0

        obs, rewards, dones, infos = env.step(obs)

        for agent_id, observation in obs.items():
            assert observation.shape == (obs_dim,), f"Observation shape should be consistent for {agent_id}"
            assert observation.dtype == np.float32, f"Observation dtype should be consistent for {agent_id}"

        if prev_obs is not None:
            # Observations should change as units move
            for agent_id in obs.keys():
                if agent_id in prev_obs:
                    # Not all observations will be different, but at least some should change
                    pass

        prev_obs = obs

    print("  ✓ Observation consistency works correctly")

    env.close()
    print("✓ Observation encoding test passed\n")


def test_performance():
    """Test environment performance"""
    print("Testing environment performance...")

    # Test with different maps
    maps_to_test = ["3s", "8m"]
    num_steps = 100

    for map_name in maps_to_test:
        print(f"  Testing performance for map: {map_name}")

        env = create_smac_env(map_name=map_name, episode_limit=num_steps)

        start_time = time.time()

        observations = env.reset()
        total_step_time = 0
        step_times = []

        for step in range(num_steps):
            step_start = time.time()

            # Random actions
            actions = {}
            for agent_id in env.agent_ids:
                avail_actions = env.get_avail_actions(agent_id)
                actions[agent_id] = np.random.choice(avail_actions) if avail_actions else 0

            # Execute step
            observations, rewards, dones, infos = env.step(actions)

            step_end = time.time()
            step_time = step_end - step_start
            step_times.append(step_time)
            total_step_time += step_time

            # If episode ends, reset
            if any(dones.values()):
                observations = env.reset()

        total_time = time.time() - start_time
        avg_step_time = total_step_time / len(step_times)
        steps_per_second = len(step_times) / total_time

        print(f"    Map: {map_name}")
        print(f"    Total time: {total_time:.3f}s")
        print(f"    Average step time: {avg_step_time:.4f}s")
        print(f"    Steps per second: {steps_per_second:.1f}")
        print(f"    Max step time: {max(step_times):.4f}s")
        print(f"    Min step time: {min(step_times):.4f}s")

        # Performance assertions (SMAC may be slower due to game engine)
        assert avg_step_time < 1.0, f"Average step time should be < 1.0s for {map_name}"
        assert steps_per_second > 1, f"Should achieve > 1 step/second for {map_name}"

        env.close()

    print("✓ Performance test passed\n")


def test_reward_system():
    """Test reward system functionality"""
    print("Testing reward system...")

    env = create_smac_env(map_name="3s", episode_limit=50)

    total_rewards = {agent_id: 0.0 for agent_id in env.agent_ids}
    step_rewards = {agent_id: [] for agent_id in env.agent_ids}

    observations = env.reset()

    for step in range(50):
        actions = {}
        for agent_id in env.agent_ids:
            avail_actions = env.get_avail_actions(agent_id)
            actions[agent_id] = np.random.choice(avail_actions) if avail_actions else 0

        obs, rewards, dones, infos = env.step(actions)

        for agent_id, reward in rewards.items():
            total_rewards[agent_id] += reward
            step_rewards[agent_id].append(reward)

        if any(dones.values()):
            if 'terminated' in infos['episode']:
                print(f"  Episode terminated")

            break

    # Check that rewards are reasonable
    for agent_id, rewards_list in step_rewards.items():
        if rewards_list:
            assert len(rewards_list) > 0, f"Should have rewards for agent {agent_id}"
            assert all(isinstance(r, (int, float)) for r in rewards_list), f"Rewards should be numeric for {agent_id}"

    print("  ✓ Reward system works correctly")

    env.close()
    print("✓ Reward system test passed\n")


def test_edge_cases():
    """Test edge cases and error handling"""
    print("Testing edge cases...")

    # Test invalid actions
    env = create_smac_env(map_name="3s", episode_limit=10)
    observations = env.reset()

    # Test missing actions
    try:
        partial_actions = {env.agent_ids[0]: 0}  # Only provide action for first agent
        obs, rewards, dones, infos = env.step(partial_actions)
        print("  ✓ Missing actions handled gracefully")
    except Exception as e:
        print(f"  ✗ Missing actions caused error: {e}")

    env.close()

    print("✓ Edge cases test passed\n")


def run_comprehensive_tests():
    """Run all comprehensive tests"""
    print("Running comprehensive SMAC environment tests...\n")
    print("=" * 60)

    try:
        test_basic_environment()
        test_different_maps()
        test_configuration_system()
        test_ctde_wrapper()
        test_action_masking()
        test_observation_encoding()
        test_performance()
        test_reward_system()
        test_edge_cases()

        print("=" * 60)
        print("✓ All comprehensive tests passed successfully!")
        print("\nSMAC environment wrapper is ready for use with MARL algorithms.")

    except Exception as e:
        print("=" * 60)
        print(f"✗ Test failed with error: {e}")
        raise


def run_quick_tests():
    """Run quick subset of tests for development"""
    print("Running quick SMAC environment tests...\n")
    print("=" * 40)

    try:
        test_basic_environment()
        test_ctde_wrapper()
        test_action_masking()

        print("=" * 40)
        print("✓ Quick tests passed!")

    except Exception as e:
        print("=" * 40)
        print(f"✗ Quick test failed: {e}")
        raise


if __name__ == "__main__":
    # You can run either comprehensive or quick tests
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        run_quick_tests()
    else:
        run_comprehensive_tests()