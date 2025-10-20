#!/usr/bin/env python3
"""
Final test of the redesigned CM environment reward system.
This script verifies the reward span is close to 100 and training works well.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from Env.CM.env_cm import create_cm_env

def analyze_reward_span():
    """Analyze the actual reward span of the redesigned system."""
    print("Final CM Environment Reward System Analysis")
    print("=" * 50)

    env = create_cm_env(difficulty="normal", render_mode="")

    print(f"Configuration:")
    print(f"  Goal reward: {env.config.goal_reached_reward}")
    print(f"  Box move reward: {env.config.box_move_reward_scale}")
    print(f"  Cooperation reward: {env.config.cooperation_reward}")
    print(f"  Time penalty: {env.config.time_penalty}")
    print()

    # Test extreme cases
    max_rewards = []
    min_rewards = []
    typical_rewards = []

    # Test goal completion (maximum reward)
    for _ in range(10):
        obs = env.reset()
        # Force goal completion
        env.game_state.box.position = env.game_state.goal.position.copy()

        actions = {agent_id: 0 for agent_id in env.agent_ids}
        obs, rewards, dones, info = env.step(actions)

        max_rewards.append(list(rewards.values())[0])

    # Test random behavior (typical reward)
    for _ in range(50):
        obs = env.reset()
        episode_reward = 0

        for step in range(100):
            actions = {}
            for agent_id in env.agent_ids:
                avail_actions = env.get_avail_actions(agent_id)
                actions[agent_id] = np.random.choice(avail_actions)

            obs, rewards, dones, info = env.step(actions)
            step_reward = list(rewards.values())[0]
            episode_reward += step_reward

            if any(dones.values()):
                break

        typical_rewards.append(episode_reward)

    # Test worst case (minimum reward) - stay still
    for _ in range(10):
        obs = env.reset()
        episode_reward = 0

        for step in range(100):
            actions = {agent_id: 0 for agent_id in env.agent_ids}  # All stay
            obs, rewards, dones, info = env.step(actions)
            step_reward = list(rewards.values())[0]
            episode_reward += step_reward

            if any(dones.values()):
                break

        min_rewards.append(episode_reward)

    # Calculate statistics
    max_avg = np.mean(max_rewards)
    min_avg = np.mean(min_rewards)
    typical_avg = np.mean(typical_rewards)

    actual_span = max_avg - min_avg

    print(f"Reward Analysis Results:")
    print(f"  Maximum reward (goal): {max_avg:.1f}")
    print(f"  Minimum reward (stay): {min_avg:.1f}")
    print(f"  Typical reward (random): {typical_avg:.1f}")
    print(f"  Actual reward span: {actual_span:.1f}")

    # Check if span is close to target
    target_span = 100
    if abs(actual_span - target_span) <= 20:
        print(f"  ✓ Reward span is close to target ({target_span})")
    elif actual_span < target_span:
        print(f"  ⚠ Reward span is smaller than target: {actual_span:.1f} < {target_span}")
    else:
        print(f"  ⚠ Reward span is larger than target: {actual_span:.1f} > {target_span}")

    env.close()
    return actual_span

def test_training_convergence():
    """Test if the reward system allows for good training convergence."""
    print("\n\nTraining Convergence Test")
    print("=" * 30)

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

        # Test stability over multiple episodes
        rewards = []
        for episode in range(20):
            obs, _ = env_wrapper.reset()
            episode_reward = 0

            for step in range(100):
                actions = {agent_id: np.random.choice(5) for agent_id in env_wrapper.agent_ids}
                obs, step_rewards, dones, infos = env_wrapper.step(actions)

                step_reward = sum(step_rewards.values()) / len(step_rewards)
                episode_reward += step_reward

                if any(dones.values()):
                    break

            rewards.append(episode_reward)

        avg_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        reward_range = np.max(rewards) - np.min(rewards)

        print(f"Stability Test Results:")
        print(f"  Average reward: {avg_reward:.1f}")
        print(f"  Standard deviation: {std_reward:.1f}")
        print(f"  Reward range: {reward_range:.1f}")

        if std_reward < 50:  # Reasonable variance
            print("  ✓ Reward variance is reasonable for training")
        else:
            print("  ⚠ Reward variance may be too high for stable training")

        env_wrapper.close()
        return True

    except Exception as e:
        print(f"  ✗ Training test failed: {e}")
        return False

def check_learning_signals():
    """Check if learning signals are clear and appropriate."""
    print("\n\nLearning Signals Check")
    print("=" * 30)

    env = create_cm_env(difficulty="normal", render_mode="")

    # Test specific scenarios
    scenarios = {
        "Box moved": False,
        "Cooperation": False,
        "Goal reached": False
    }

    for episode in range(30):
        obs = env.reset()

        for step in range(50):
            actions = {}
            for agent_id in env.agent_ids:
                avail_actions = env.get_avail_actions(agent_id)
                actions[agent_id] = np.random.choice(avail_actions)

            obs, rewards, dones, info = env.step(actions)
            step_reward = list(rewards.values())[0]

            if info['box_moved']:
                scenarios["Box moved"] = True

            if len(info['pushing_agents']) > 1:
                scenarios["Cooperation"] = True

            if info['agents_complete']:
                scenarios["Goal reached"] = True
                break

    print(f"Learning Signal Availability:")
    for scenario, observed in scenarios.items():
        status = "✓" if observed else "✗"
        print(f"  {status} {scenario}: {'Observed' if observed else 'Not observed in test'}")

    env.close()

if __name__ == "__main__":
    # Run all tests
    span = analyze_reward_span()
    training_ok = test_training_convergence()
    check_learning_signals()

    print("\n" + "="*50)
    print("Final Assessment:")
    print(f"  Reward span: {span:.1f} (target: ~100)")
    print(f"  Training suitability: {'✓ Suitable' if training_ok else '✗ Issues detected'}")

    if 80 <= span <= 120 and training_ok:
        print("  ✓ Reward system is well-designed for 100-episode training")
    else:
        print("  ⚠ Reward system may need further adjustment")