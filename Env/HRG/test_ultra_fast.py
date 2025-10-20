#!/usr/bin/env python3
"""
Test script for HRG Ultra-Fast Environment
Performance and functionality verification
"""

import time
import numpy as np
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from .env_hrg_ultra_fast import create_hrg_ultra_fast_env
    from .env_hrg_ultra_fast_ctde import create_hrg_ultra_fast_ctde_env
except ImportError:
    # Fallback for direct execution
    from env_hrg_ultra_fast import create_hrg_ultra_fast_env
    from env_hrg_ultra_fast_ctde import create_hrg_ultra_fast_ctde_env


def test_basic_functionality():
    """Test basic environment functionality"""
    print("Testing Basic Ultra-Fast HRG Environment...")

    # Create environment
    env = create_hrg_ultra_fast_env()

    # Get environment info
    print(f"Number of agents: {env.n_agents}")
    print(f"Agent IDs: {env.agent_ids}")
    print(f"Observation dimensions: {env.obs_dims}")
    print(f"Action dimensions: {env.act_dims}")
    print(f"Max steps: {env.config.max_steps}")
    print(f"Grid size: {env.config.grid_size}")

    # Test reset
    obs = env.reset()
    print(f"Reset successful. Observations shape: {[o.shape for o in obs.values()]}")

    # Test a few steps
    total_reward = 0
    start_time = time.time()

    for step in range(20):
        # Random actions
        actions = {agent_id: np.random.randint(0, 8) for agent_id in env.agent_ids}

        # Step environment
        observations, rewards, dones, infos = env.step(actions)
        total_reward += sum(rewards.values())

        if step % 5 == 0:
            print(f"Step {step}: Total reward = {total_reward:.2f}")

    end_time = time.time()
    print(f"20 steps completed in {end_time - start_time:.4f} seconds")
    print(f"Steps per second: {20 / (end_time - start_time):.1f}")
    print("Basic functionality test passed!\n")


def test_ctde_functionality():
    """Test CTDE environment functionality"""
    print("Testing Ultra-Fast HRG CTDE Environment...")

    # Create CTDE environment
    env = create_hrg_ultra_fast_ctde_env()

    # Test reset
    obs, info = env.reset()
    print(f"CTDE reset successful.")
    print(f"Global state shape: {info['global_state'].shape}")

    # Test a few steps
    total_reward = 0
    start_time = time.time()

    for step in range(20):
        # Random actions
        actions = {agent_id: np.random.randint(0, 8) for agent_id in env.agent_ids}

        # Step environment
        observations, rewards, done, truncated, info = env.step(actions)
        total_reward += sum(rewards.values())

        if step % 5 == 0:
            print(f"Step {step}: Total reward = {total_reward:.2f}, "
                  f"Global state shape = {info['global_state'].shape}")

    end_time = time.time()
    print(f"20 CTDE steps completed in {end_time - start_time:.4f} seconds")
    print(f"CTDE Steps per second: {20 / (end_time - start_time):.1f}")
    print("CTDE functionality test passed!\n")


def benchmark_performance():
    """Benchmark environment performance"""
    print("Performance Benchmark...")

    # Test different environments
    configs = [
        ("Ultra-Fast", create_hrg_ultra_fast_env()),
        ("Ultra-Fast CTDE", create_hrg_ultra_fast_ctde_env())
    ]

    num_episodes = 5
    steps_per_episode = 100

    for name, env in configs:
        print(f"\nBenchmarking {name}...")
        total_time = 0
        total_steps = 0

        for episode in range(num_episodes):
            if isinstance(env, type(create_hrg_ultra_fast_ctde_env())):
                obs, info = env.reset()
            else:
                obs = env.reset()

            episode_start = time.time()

            for step in range(steps_per_episode):
                # Random actions
                actions = {agent_id: np.random.randint(0, 8) for agent_id in env.agent_ids}

                # Step environment
                if isinstance(env, type(create_hrg_ultra_fast_ctde_env())):
                    observations, rewards, done, truncated, info = env.step(actions)
                else:
                    observations, rewards, dones, infos = env.step(actions)

                if isinstance(env, type(create_hrg_ultra_fast_ctde_env())) and done:
                    break
                elif not isinstance(env, type(create_hrg_ultra_fast_ctde_env())) and all(dones.values()):
                    break

            episode_time = time.time() - episode_start
            total_time += episode_time
            total_steps += step + 1

            print(f"Episode {episode + 1}: {step + 1} steps in {episode_time:.4f}s")

        avg_time_per_step = total_time / total_steps
        steps_per_second = total_steps / total_time

        print(f"{name} Results:")
        print(f"  Total episodes: {num_episodes}")
        print(f"  Total steps: {total_steps}")
        print(f"  Average time per step: {avg_time_per_step*1000:.3f} ms")
        print(f"  Steps per second: {steps_per_second:.1f}")
        print(f"  Time per 1000 steps: {avg_time_per_step*1000:.3f} s")

    print("\nPerformance benchmark completed!\n")


def test_memory_usage():
    """Test memory efficiency"""
    print("Testing Memory Usage...")

    env = create_hrg_ultra_fast_env()

    # Run many steps to check for memory leaks
    obs = env.reset()

    initial_memory = 0  # Could use psutil for actual memory measurement

    for step in range(1000):
        actions = {agent_id: np.random.randint(0, 8) for agent_id in env.agent_ids}
        observations, rewards, dones, infos = env.step(actions)

        if all(dones.values()):
            obs = env.reset()

        if step % 200 == 0:
            print(f"Completed {step} steps...")

    print("Memory usage test completed (1000 steps)\n")


def test_agent_configurations():
    """Test different agent configurations"""
    print("Testing Agent Configurations...")

    # Test default ultra-fast config (2 agents)
    env1 = create_hrg_ultra_fast_env()
    print(f"Default ultra-fast: {env1.n_agents} agents - {env1.agent_ids}")

    # Test custom configurations
    env2 = create_hrg_ultra_fast_env(grid_size=4, max_steps=50)
    print(f"Custom config: {env2.n_agents} agents, grid {env2.config.grid_size}x{env2.config.grid_size}")

    # Test observation sizes
    obs1 = env1.reset()
    obs2 = env2.reset()

    for agent_id in env1.agent_ids:
        print(f"Agent {agent_id} observation size: {obs1[agent_id].shape}")

    print("Agent configuration tests passed!\n")


if __name__ == "__main__":
    print("=" * 60)
    print("HRG Ultra-Fast Environment Test Suite")
    print("=" * 60)

    try:
        test_basic_functionality()
        test_ctde_functionality()
        benchmark_performance()
        test_memory_usage()
        test_agent_configurations()

        print("=" * 60)
        print("All tests completed successfully!")
        print("Ultra-fast HRG environment is ready for training.")
        print("=" * 60)

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()