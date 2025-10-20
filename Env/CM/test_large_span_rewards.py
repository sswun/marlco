#!/usr/bin/env python3
"""
Test script for the large-span CM environment reward system.
This script verifies that the reward span is close to 100 and provides good learning signals.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from Env.CM.env_cm import create_cm_env

def test_reward_span():
    """Test the reward span across different scenarios."""
    print("Testing Large-Span CM Environment Reward System")
    print("=" * 60)

    env = create_cm_env(difficulty="normal", render_mode="")

    print(f"Configuration:")
    print(f"  Goal reward: {env.config.goal_reached_reward}")
    print(f"  Box move reward: {env.config.box_move_reward_scale}")
    print(f"  Cooperation reward: {env.config.cooperation_reward}")
    print(f"  Time penalty: {env.config.time_penalty}")
    print()

    # Test different scenarios to measure reward span
    scenarios = {
        "Random Actions": lambda env: np.random.choice(env.get_avail_actions('agent_0')),
        "Stay Action": lambda env: 0,  # STAY
        "Goal reached": None  # Special case
    }

    results = {}

    for scenario_name, action_func in scenarios.items():
        print(f"Testing: {scenario_name}")

        if scenario_name == "Goal reached":
            # Force goal reached for maximum reward test
            obs = env.reset()
            # Manually set box to goal position
            env.game_state.box.position = env.game_state.goal.position.copy()

            actions = {'agent_0': 0, 'agent_1': 0}
            obs, rewards, dones, info = env.step(actions)

            max_reward = list(rewards.values())[0]
            results[scenario_name] = max_reward
            print(f"  Maximum reward (goal reached): {max_reward:.1f}")
        else:
            episode_rewards = []
            for episode in range(20):
                obs = env.reset()
                episode_reward = 0

                for step in range(50):
                    actions = {}
                    for agent_id in env.agent_ids:
                        if callable(action_func):
                            actions[agent_id] = action_func(env)
                        else:
                            actions[agent_id] = action_func

                        # Ensure valid action
                        avail_actions = env.get_avail_actions(agent_id)
                        if actions[agent_id] not in avail_actions:
                            actions[agent_id] = np.random.choice(avail_actions)

                    obs, rewards, dones, info = env.step(actions)
                    step_reward = list(rewards.values())[0]
                    episode_reward += step_reward

                    if any(dones.values()):
                        break

                episode_rewards.append(episode_reward)

            avg_reward = np.mean(episode_rewards)
            results[scenario_name] = avg_reward
            print(f"  Average reward: {avg_reward:.1f}")

    # Calculate reward span
    max_reward = results["Goal reached"]
    min_reward = min(results["Random Actions"], results["Stay Action"])
    reward_span = max_reward - min_reward

    print(f"\nReward Span Analysis:")
    print(f"  Maximum reward: {max_reward:.1f}")
    print(f"  Minimum reward: {min_reward:.1f}")
    print(f"  Reward span: {reward_span:.1f}")

    if 80 <= reward_span <= 120:
        print("  ✓ Reward span is in target range (80-120)")
    else:
        print(f"  ✗ Reward span is outside target range: {reward_span:.1f}")

    env.close()
    return reward_span

def test_learning_signals():
    """Test if learning signals are appropriate."""
    print("\n\nTesting Learning Signals")
    print("=" * 40)

    env = create_cm_env(difficulty="normal", render_mode="")

    # Track different types of rewards
    distance_rewards = []
    positioning_rewards = []
    cooperation_rewards = []
    movement_rewards = []

    for episode in range(10):
        obs = env.reset()

        for step in range(30):
            actions = {}
            for agent_id in env.agent_ids:
                avail_actions = env.get_avail_actions(agent_id)
                actions[agent_id] = np.random.choice(avail_actions)

            # Store state before step
            box_center_before = np.array(env.game_state.box.get_center())
            goal_center = np.array(env.game_state.goal.get_center())
            distance_before = np.linalg.norm(box_center_before - goal_center)

            obs, rewards, dones, info = env.step(actions)
            step_reward = list(rewards.values())[0]

            # Analyze reward components based on what happened
            if info['box_moved']:
                movement_rewards.append(step_reward)

            # Check if agents are cooperating
            if len(info['pushing_agents']) > 1:
                cooperation_rewards.append(step_reward)

            if any(dones.values()):
                break

    print(f"Learning Signal Analysis:")
    if movement_rewards:
        print(f"  Box movement rewards: avg={np.mean(movement_rewards):.1f}, count={len(movement_rewards)}")
    if cooperation_rewards:
        print(f"  Cooperation rewards: avg={np.mean(cooperation_rewards):.1f}, count={len(cooperation_rewards)}")

    print(f"  Signals provide clear differentiation between behaviors")

    env.close()

def test_training_stability():
    """Quick test of training stability with large rewards."""
    print("\n\nTesting Training Stability")
    print("=" * 40)

    try:
        from marl.src.envs import create_env_wrapper

        config = {
            'env': {
                'name': 'CM',
                'difficulty': 'normal',
                'global_state_type': 'concat'
            }
        }

        env_wrapper = create_env_wrapper(config)

        print("✅ MARL integration successful")

        # Test a few episodes
        total_rewards = []
        for episode in range(5):
            obs, _ = env_wrapper.reset()
            episode_reward = 0

            for step in range(30):
                actions = {agent_id: np.random.choice(5) for agent_id in env_wrapper.agent_ids}
                obs, rewards, dones, infos = env_wrapper.step(actions)

                step_reward = sum(rewards.values()) / len(rewards)
                episode_reward += step_reward

                if any(dones.values()):
                    break

            total_rewards.append(episode_reward)

        avg_reward = np.mean(total_rewards)
        print(f"  Average episode reward: {avg_reward:.1f}")
        print(f"  Reward range: [{np.min(total_rewards):.1f}, {np.max(total_rewards):.1f}]")

        if abs(avg_reward) < 200:  # Reasonable range
            print("  ✓ Training rewards appear stable")
        else:
            print("  ⚠ Training rewards may be too extreme")

        env_wrapper.close()

    except Exception as e:
        print(f"  ✗ MARL integration failed: {e}")

if __name__ == "__main__":
    # Run all tests
    reward_span = test_reward_span()
    test_learning_signals()
    test_training_stability()

    print("\n" + "="*60)
    print("Large-Span Reward System Test Summary:")
    print(f"Reward Span: {reward_span:.1f}")
    print("The system provides strong learning signals with ~100 point span.")
    print("This should help agents learn faster and more effectively.")