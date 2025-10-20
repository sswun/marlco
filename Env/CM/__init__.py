"""
CM (Collaborative Moving) Environment Package

A multi-agent collaborative box pushing environment designed for testing
multi-agent reinforcement learning algorithms. The environment is simple yet
effective for verifying algorithm convergence.

Key Features:
- Multi-agent cooperation required for optimal performance
- Simple action and observation spaces
- CTDE (Centralized Training Decentralized Execution) compatible
- Multiple difficulty levels and configurations
- Rich visualization capabilities
- Comprehensive testing suite

Example Usage:
    from Env.CM import create_cm_env, create_cm_ctde_env

    # Basic environment
    env = create_cm_env(difficulty="easy", render_mode="human")
    obs, info = env.reset()
    actions = {agent_id: env.get_avail_actions(agent_id)[0] for agent_id in env.agent_ids}
    obs, rewards, terminated, truncated, info = env.step(actions)

    # CTDE environment for MARL algorithms
    ctde_env = create_cm_ctde_env(difficulty="easy_ctde", global_state_type="concat")
    obs, global_state = ctde_env.reset()
    obs, rewards, terminated, truncated, info = ctde_env.step(actions)
"""

from .core import (
    Position,
    Box,
    Goal,
    Agent,
    CMGameState,
    ActionType
)

from .config import (
    CMConfig,
    get_easy_config,
    get_normal_config,
    get_hard_config,
    get_debug_config,
    get_easy_ctde_config,
    get_normal_ctde_config,
    get_hard_ctde_config,
    get_config_by_name,
    list_available_configs
)

from .env_cm import (
    CooperativeMovingEnv,
    create_cm_env,
    create_cm_env_from_config
)

from .env_cm_ctde import (
    CooperativeMovingCTDEEnv,
    CMCTDEWrapper,
    CMEnvironmentAdapter,
    create_cm_ctde_env,
    wrap_cm_env_to_ctde,
    create_cm_adapter
)

from .renderer import (
    MatplotlibRenderer,
    AnimationRenderer,
    TextRenderer,
    create_visualization_sequence
)

# Version information
__version__ = "1.0.0"
__author__ = "MARL Research Team"

# Export main classes and functions
__all__ = [
    # Core classes
    'Position',
    'Box',
    'Goal',
    'Agent',
    'CMGameState',
    'ActionType',

    # Configuration
    'CMConfig',
    'get_easy_config',
    'get_normal_config',
    'get_hard_config',
    'get_debug_config',
    'get_easy_ctde_config',
    'get_normal_ctde_config',
    'get_hard_ctde_config',
    'get_config_by_name',
    'list_available_configs',

    # Environment classes
    'CooperativeMovingEnv',
    'create_cm_env',
    'create_cm_env_from_config',

    # CTDE classes
    'CooperativeMovingCTDEEnv',
    'CMCTDEWrapper',
    'CMEnvironmentAdapter',
    'create_cm_ctde_env',
    'wrap_cm_env_to_ctde',
    'create_cm_adapter',

    # Rendering
    'MatplotlibRenderer',
    'AnimationRenderer',
    'TextRenderer',
    'create_visualization_sequence'
]


def get_package_info():
    """Get package information."""
    return {
        'name': 'CM Environment',
        'version': __version__,
        'description': 'Multi-agent collaborative box pushing environment',
        'features': [
            'Multi-agent cooperation mechanics',
            'Simple action and observation spaces',
            'CTDE compatibility for MARL algorithms',
            'Multiple difficulty levels',
            'Rich visualization capabilities',
            'Comprehensive testing suite'
        ],
        'supported_algorithms': [
            'QMIX', 'VDN', 'IQL', 'MADDPG', 'MAPPO', 'Other CTDE algorithms'
        ],
        'environment_types': [
            'Cooperative task',
            'Discrete actions',
            'Partial observability',
            'Team-based rewards'
        ]
    }


def quick_test():
    """Run a quick test to verify the package works."""
    try:
        print("Testing CM Environment Package...")

        # Test basic environment creation
        env = create_cm_env(difficulty="debug")
        obs, info = env.reset()

        # Test CTDE environment creation
        ctde_env = create_cm_ctde_env(difficulty="debug", global_state_type="concat")
        obs, global_state = ctde_env.reset()

        # Test configuration system
        configs = list_available_configs()
        config = get_config_by_name("easy")

        env.close()
        ctde_env.close()

        print(f"✅ CM Environment Package v{__version__} is working correctly!")
        print(f"   Available configurations: {configs}")
        print(f"   Environment info: {env.get_env_info()}")

        return True

    except Exception as e:
        print(f"❌ Package test failed: {e}")
        return False


def demo_environment():
    """Run a demonstration of the CM environment."""
    print("Running CM Environment Demo...")

    # Create environment with visualization
    env = create_cm_env(difficulty="easy", render_mode="human")

    print(f"Environment Info:")
    info = env.get_env_info()
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Reset environment
    obs, reset_info = env.reset()
    print(f"\nInitial observations: {list(obs.keys())}")

    # Run a few episodes with different strategies
    for episode in range(2):
        print(f"\n--- Episode {episode + 1} ---")
        obs, info = env.reset()
        total_reward = 0
        step = 0

        while step < 20:  # Limit for demo
            # Random actions
            actions = {}
            for agent_id in env.agent_ids:
                avail_actions = env.get_avail_actions(agent_id)
                actions[agent_id] = np.random.choice(avail_actions)

            # Execute step
            obs, rewards, terminated, truncated, info = env.step(actions)

            # Accumulate reward
            episode_reward = sum(rewards.values())
            total_reward += episode_reward

            step += 1
            print(f"  Step {step}: reward={episode_reward:.3f}, distance={info['distance_to_goal']:.2f}")

            if terminated:
                print(f"  🎯 Goal reached in {step} steps!")
                break
            elif truncated:
                print(f"  ⏱️ Episode truncated after {step} steps")
                break

        print(f"  Episode total reward: {total_reward:.3f}")

    env.close()
    print("\nDemo completed!")


if __name__ == "__main__":
    import numpy as np

    print("=" * 60)
    print("CM (Collaborative Moving) Environment")
    print("=" * 60)

    # Show package info
    pkg_info = get_package_info()
    print(f"Name: {pkg_info['name']}")
    print(f"Version: {pkg_info['version']}")
    print(f"Description: {pkg_info['description']}")
    print("\nKey Features:")
    for feature in pkg_info['features']:
        print(f"  • {feature}")

    print("\nSupported MARL Algorithms:")
    for algo in pkg_info['supported_algorithms']:
        print(f"  • {algo}")

    # Run quick test
    print("\n" + "=" * 60)
    if quick_test():
        print("\n" + "=" * 60)
        demo_environment()

    print("\n" + "=" * 60)
    print("CM Environment Package ready for use!")
    print("For more examples, see CM_Tutorial.ipynb")
    print("=" * 60)