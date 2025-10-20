#!/usr/bin/env python3
"""
Detailed analysis of the CM environment reward system.
This script analyzes reward distributions and patterns to ensure proper scaling.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from Env.CM.env_cm import create_cm_env
import matplotlib.pyplot as plt

def analyze_reward_distribution():
    """Analyze the distribution of rewards across different scenarios."""
    print("CM Environment Reward Distribution Analysis")
    print("=" * 50)

    env = create_cm_env(difficulty="easy", render_mode="")

    # Test scenarios
    scenarios = [
        ("Random Actions", lambda env: [np.random.choice(env.get_avail_actions(agent_id))
                                       for agent_id in env.agent_ids]),
        ("Stay Actions", lambda env: [0 for _ in env.agent_ids]),  # All stay
        ("Toward Box", lambda env: [get_toward_box_action(env, agent_id)
                                    for agent_id in env.agent_ids])
    ]

    results = {}

    for scenario_name, action_func in scenarios:
        print(f"\nTesting: {scenario_name}")
        print("-" * 30)

        episode_rewards = []
        step_rewards = []
        box_moved_count = 0
        goal_reached_count = 0

        for episode in range(20):  # Test episodes
            obs = env.reset()
            episode_reward = 0

            for step in range(50):  # Short episodes
                actions = {}
                for i, agent_id in enumerate(env.agent_ids):
                    actions[agent_id] = action_func(env)[i]

                obs, rewards, dones, info = env.step(actions)
                step_reward = list(rewards.values())[0]
                episode_reward += step_reward
                step_rewards.append(step_reward)

                if info['box_moved']:
                    box_moved_count += 1

                if info['agents_complete']:
                    goal_reached_count += 1
                    break

            episode_rewards.append(episode_reward)

        # Calculate statistics
        avg_episode_reward = np.mean(episode_rewards)
        std_episode_reward = np.std(episode_rewards)
        avg_step_reward = np.mean(step_rewards)
        box_moved_rate = box_moved_count / (20 * 50)
        goal_rate = goal_reached_count / 20

        results[scenario_name] = {
            'avg_episode_reward': avg_episode_reward,
            'std_episode_reward': std_episode_reward,
            'avg_step_reward': avg_step_reward,
            'box_moved_rate': box_moved_rate,
            'goal_rate': goal_rate,
            'step_rewards': step_rewards
        }

        print(f"  Average episode reward: {avg_episode_reward:.3f} ± {std_episode_reward:.3f}")
        print(f"  Average step reward: {avg_step_reward:.3f}")
        print(f"  Box movement rate: {box_moved_rate:.3f}")
        print(f"  Goal reached rate: {goal_rate:.3f}")

    # Compare scenarios
    print(f"\nScenario Comparison:")
    print("-" * 30)
    for name, stats in results.items():
        print(f"{name:15s}: {stats['avg_episode_reward']:6.3f} per episode, "
              f"{stats['avg_step_reward']:6.3f} per step")

    env.close()
    return results

def get_toward_box_action(env, agent_id):
    """Get action that moves agent toward the box center."""
    agent = env.game_state.get_agent(agent_id)
    box_center = env.game_state.box.get_center()

    # Calculate direction to box
    dx = box_center[0] - agent.position.x
    dy = box_center[1] - agent.position.y

    # Choose action that moves toward box
    if abs(dx) > abs(dy):
        if dx > 0:
            return 1  # DOWN
        else:
            return 2  # UP
    else:
        if dy > 0:
            return 3  # RIGHT
        else:
            return 4  # LEFT

def analyze_reward_components():
    """Analyze individual reward components."""
    print("\n\nDetailed Reward Component Analysis")
    print("=" * 50)

    env = create_cm_env(difficulty="easy", render_mode="")
    obs = env.reset()

    # Track different reward components
    components = {
        'time_penalty': [],
        'distance_improvement': [],
        'positioning': [],
        'cooperation': [],
        'box_movement': [],
        'goal_bonus': []
    }

    for step in range(100):
        # Random actions
        actions = {}
        for agent_id in env.agent_ids:
            avail_actions = env.get_avail_actions(agent_id)
            actions[agent_id] = np.random.choice(avail_actions)

        # Manually calculate each component
        collision_penalty = 0.0  # Simplified

        # Time penalty
        time_penalty = env.config.time_penalty + collision_penalty
        components['time_penalty'].append(time_penalty)

        # Distance improvement
        box_center = np.array(env.game_state.box.get_center())
        goal_center = np.array(env.game_state.goal.get_center())
        current_distance = np.linalg.norm(box_center - goal_center)

        if hasattr(env, '_last_box_distance') and env._last_box_distance is not None:
            distance_improvement = env._last_box_distance - current_distance
            if distance_improvement > 0.1:
                distance_reward = distance_improvement * 0.1
            else:
                distance_reward = 0.0
        else:
            distance_reward = 0.0
        components['distance_improvement'].append(distance_reward)

        # Positioning reward
        agents_at_box = 0
        for agent_id, agent in env.game_state.agents.items():
            agent_pos = np.array([agent.position.x, agent.position.y])
            distance_to_box = np.linalg.norm(agent_pos - box_center)
            if distance_to_box <= 1.5:
                agents_at_box += 1

        positioning_reward = agents_at_box * 0.02 if agents_at_box > 0 else 0.0
        components['positioning'].append(positioning_reward)

        # Cooperation reward
        pushing_sides = env.game_state.get_push_sides()
        n_pushing = len(pushing_sides)

        if n_pushing > 1:
            cooperation_bonus = env.config.cooperation_reward * n_pushing
            if n_pushing >= 2:
                unique_sides = len(set(pushing_sides))
                if unique_sides >= 2:
                    cooperation_bonus += 0.01
        else:
            cooperation_bonus = 0.0
        components['cooperation'].append(cooperation_bonus)

        # Box movement reward
        obs, rewards, dones, info = env.step(actions)
        box_moved = info['box_moved']
        box_movement_reward = env.config.box_move_reward_scale if box_moved else 0.0
        components['box_movement'].append(box_movement_reward)

        # Goal bonus
        goal_bonus = env.config.goal_reached_reward if info['agents_complete'] else 0.0
        if info['agents_complete'] and env.current_step < env.config.max_steps * 0.5:
            goal_bonus += 1.0
        components['goal_bonus'].append(goal_bonus)

        if info['agents_complete']:
            break

    # Print component statistics
    print("Reward Component Statistics:")
    for component, values in components.items():
        if values:  # Check if list is not empty
            avg_val = np.mean(values)
            max_val = np.max(values)
            non_zero_count = sum(1 for v in values if abs(v) > 1e-6)
            print(f"  {component:15s}: avg={avg_val:6.4f}, max={max_val:6.4f}, "
                  f"non-zero={non_zero_count:3d}/{len(values)}")

    env.close()

def check_reward_scale_reasonableness():
    """Check if reward scales are reasonable for learning."""
    print("\n\nReward Scale Reasonableness Check")
    print("=" * 50)

    env = create_cm_env(difficulty="easy", render_mode="")

    # Collect reward statistics over many random episodes
    all_rewards = []
    episode_lengths = []
    goal_rewards = []

    for episode in range(50):
        obs = env.reset()
        episode_reward = 0
        step_count = 0

        for step in range(100):
            actions = {}
            for agent_id in env.agent_ids:
                avail_actions = env.get_avail_actions(agent_id)
                actions[agent_id] = np.random.choice(avail_actions)

            obs, rewards, dones, info = env.step(actions)
            step_reward = list(rewards.values())[0]
            episode_reward += step_reward
            step_count += 1

            if info['agents_complete']:
                goal_rewards.append(episode_reward)
                break

        all_rewards.append(episode_reward)
        episode_lengths.append(step_count)

    # Analysis
    avg_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    max_reward = np.max(all_rewards)
    min_reward = np.min(all_rewards)

    avg_length = np.mean(episode_lengths)
    goal_rate = len(goal_rewards) / 50

    print(f"Random Performance (50 episodes):")
    print(f"  Average reward: {avg_reward:.3f} ± {std_reward:.3f}")
    print(f"  Reward range: [{min_reward:.3f}, {max_reward:.3f}]")
    print(f"  Average episode length: {avg_length:.1f}")
    print(f"  Goal reached rate: {goal_rate:.2f}")

    # Reasonableness checks
    print(f"\nReasonableness Checks:")

    # Check 1: Random rewards should be small (close to time penalty)
    expected_random_reward = env.config.time_penalty * avg_length
    print(f"  Random vs Expected: {avg_reward:.3f} vs {expected_random_reward:.3f}")
    if abs(avg_reward - expected_random_reward) < 5.0:
        print("  ✓ Random rewards are reasonably small")
    else:
        print("  ✗ Random rewards may be too large")

    # Check 2: Goal rewards should be significantly larger
    if goal_rewards:
        avg_goal_reward = np.mean(goal_rewards)
        print(f"  Goal completion reward: {avg_goal_reward:.3f}")
        if avg_goal_reward > avg_reward * 2:
            print("  ✓ Goal rewards provide strong positive signal")
        else:
            print("  ✗ Goal rewards may not be sufficiently distinctive")

    # Check 3: No extreme values
    if abs(max_reward) < 50 and abs(min_reward) < 10:
        print("  ✓ Reward values are in reasonable range")
    else:
        print("  ✗ Reward values may be too extreme")

    env.close()

if __name__ == "__main__":
    # Run all analyses
    results = analyze_reward_distribution()
    analyze_reward_components()
    check_reward_scale_reasonableness()

    print("\n\n" + "="*50)
    print("Reward analysis completed!")
    print("If all checks pass, the reward system should be well-balanced for learning.")