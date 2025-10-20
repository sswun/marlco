#!/usr/bin/env python3
"""
Test script for the enhanced CM environment reward system.
This script demonstrates the improved reward mechanism with better learning signals.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from Env.CM.env_cm import create_cm_env

def test_enhanced_rewards():
    """Test the enhanced reward system."""
    print("Testing Enhanced CM Environment Reward System")
    print("=" * 50)

    # Create environment
    env = create_cm_env(difficulty="easy", render_mode="")

    print(f"Environment Info:")
    print(f"  Grid size: {env.config.grid_size}")
    print(f"  Number of agents: {env.n_agents}")
    print(f"  Max steps: {env.config.max_steps}")
    print(f"  Box size: {env.config.box_size}")
    print(f"  Goal size: {env.config.goal_size}")
    print()

    # Run multiple episodes to test reward system
    n_episodes = 3
    total_rewards = []

    for episode in range(n_episodes):
        print(f"Episode {episode + 1}:")
        print("-" * 30)

        obs = env.reset()
        episode_reward = 0
        step_rewards = []

        # Get initial state
        info = env._get_info()
        print(f"  Initial distance to goal: {info['distance_to_goal']:.2f}")

        for step in range(20):  # Short test episodes
            # Generate random actions
            actions = {}
            for agent_id in env.agent_ids:
                avail_actions = env.get_avail_actions(agent_id)
                actions[agent_id] = np.random.choice(avail_actions)

            # Execute step
            obs, rewards, dones, info = env.step(actions)

            # Track rewards
            step_reward = list(rewards.values())[0]
            episode_reward += step_reward
            step_rewards.append(step_reward)

            print(f"  Step {step + 1:2d}: reward={step_reward:6.3f}, "
                  f"distance={info['distance_to_goal']:.2f}, "
                  f"box_moved={info['box_moved']}, "
                  f"pushing_agents={len(info['pushing_agents'])}")

            if any(dones.values()):
                print(f"  Episode completed at step {step + 1}!")
                break

        total_rewards.append(episode_reward)
        print(f"  Episode total reward: {episode_reward:.3f}")
        print()

    print("Summary:")
    print(f"  Average episode reward: {np.mean(total_rewards):.3f}")
    print(f"  Min episode reward: {np.min(total_rewards):.3f}")
    print(f"  Max episode reward: {np.max(total_rewards):.3f}")
    print()

    # Test specific reward components
    print("Testing Specific Reward Components:")
    print("-" * 40)

    obs = env.reset()

    # Test movement toward box
    print("1. Testing agent positioning rewards...")
    for step in range(5):
        # Get agents to move toward box
        actions = {}
        for i, agent_id in enumerate(env.agent_ids):
            # Simple heuristic: move toward box center
            agent = env.game_state.get_agent(agent_id)
            box_center = env.game_state.box.get_center()

            # Choose action that moves toward box
            dx = box_center[0] - agent.position.x
            dy = box_center[1] - agent.position.y

            if abs(dx) > abs(dy):
                action = 1 if dx > 0 else 2  # DOWN or UP
            else:
                action = 3 if dy > 0 else 4  # RIGHT or LEFT

            actions[agent_id] = action

        obs, rewards, dones, info = env.step(actions)
        step_reward = list(rewards.values())[0]
        print(f"  Step {step + 1}: reward={step_reward:6.3f}, "
              f"distance_to_goal={info['distance_to_goal']:.2f}")

    print()
    print("Enhanced reward system test completed successfully!")
    env.close()

def test_reward_components():
    """Test individual reward components in detail."""
    print("\nTesting Individual Reward Components:")
    print("=" * 50)

    env = create_cm_env(difficulty="easy", render_mode="")
    obs = env.reset()

    # Get initial information
    info = env._get_info()
    print(f"Initial state:")
    print(f"  Box position: {info['box_position']}")
    print(f"  Goal position: {info['goal_position']}")
    print(f"  Distance to goal: {info['distance_to_goal']:.2f}")
    print()

    # Test distance improvement rewards
    print("Testing distance improvement rewards:")

    # Force a move that should improve distance
    for step in range(3):
        actions = {}
        for agent_id in env.agent_ids:
            # Try different random actions
            avail_actions = env.get_avail_actions(agent_id)
            actions[agent_id] = np.random.choice(avail_actions)

        obs, rewards, dones, info = env.step(actions)
        step_reward = list(rewards.values())[0]

        print(f"  Step {step + 1}: reward={step_reward:6.3f}, "
              f"distance={info['distance_to_goal']:.2f}, "
              f"box_moved={info['box_moved']}")

    env.close()

if __name__ == "__main__":
    test_enhanced_rewards()
    test_reward_components()