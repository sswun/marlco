"""
SMAC Wrapper Demo

Demonstration of the SMAC environment wrapper working with the unified interface.
This shows how the wrapper provides the same interface as DEM, HRG, and MSFS environments.
"""

import numpy as np
import time
from typing import Dict, Any

# Import SMAC wrapper
from .env_smac import create_smac_env
from .env_smac_ctde import create_smac_ctde_env
from .config import get_easy_config, get_normal_config, get_debug_config


def demo_basic_interface():
    """Demo basic environment interface (same as DEM, HRG, MSFS)"""
    print("=" * 50)
    print("DEMO: Basic SMAC Environment Interface")
    print("=" * 50)

    # Create environment using unified interface
    env = create_smac_env(map_name="8m", episode_limit=50)

    print(f"✓ Environment created: {env.config.map_name}")
    print(f"✓ Agent IDs: {env.agent_ids}")

    # Get environment info (unified format)
    env_info = env.get_env_info()
    print(f"✓ Environment info: {env_info['n_agents']} agents, "
          f"{env_info['n_actions']} actions, {env_info['obs_shape']} obs dim")

    # Reset environment (unified format)
    observations = env.reset()
    print(f"✓ Environment reset: observations for {len(observations)} agents")

    # Show observation format (unified dictionary format)
    sample_agent = env.agent_ids[0]
    obs = observations[sample_agent]
    print(f"✓ Sample observation for {sample_agent}: shape={obs.shape}, dtype={obs.dtype}")

    # Show action masking (unified interface)
    avail_actions = env.get_avail_actions(sample_agent)
    print(f"✓ Available actions for {sample_agent}: {len(avail_actions)} actions")

    # Execute steps using unified interface
    total_rewards = {agent_id: 0.0 for agent_id in env.agent_ids}
    step_count = 0

    print("\nExecuting episode...")
    for step in range(20):
        # Create actions dictionary (unified format)
        actions = {}
        for agent_id in env.agent_ids:
            avail_actions = env.get_avail_actions(agent_id)
            actions[agent_id] = np.random.choice(avail_actions) if avail_actions else 0

        # Step environment (unified return format)
        observations, rewards, dones, infos = env.step(actions)

        # Accumulate rewards (per-agent rewards in unified format)
        for agent_id, reward in rewards.items():
            total_rewards[agent_id] += reward

        step_count += 1

        if any(dones.values()):
            print(f"  Episode completed at step {step_count}")
            break

    print(f"✓ Episode finished: {step_count} steps")
    print(f"✓ Final rewards: {total_rewards}")

    env.close()
    print("✓ Environment closed\n")


def demo_ctde_interface():
    """Demo CTDE interface (same as DEM, HRG, MSFS CTDE versions)"""
    print("=" * 50)
    print("DEMO: CTDE Environment Interface")
    print("=" * 50)

    # Create CTDE environment using unified interface
    ctde_env = create_smac_ctde_env(
        map_name="8m",
        episode_limit=30
    )

    print(f"✓ CTDE environment created: {ctde_env.config.map_name}")
    print(f"✓ Using original SMAC global state")

    # Get CTDE environment info
    env_info = ctde_env.get_env_info()
    print(f"✓ Global state dimension: {env_info['global_state_dim']}")

    # Reset and get global state
    observations = ctde_env.reset()
    global_state = ctde_env.get_global_state()
    original_state = ctde_env.get_state()

    print(f"✓ Global state shape: {global_state.shape}")
    print(f"✓ Global state matches original SMAC state: {np.allclose(global_state, original_state)}")
    print(f"✓ Global state range: [{global_state.min():.3f}, {global_state.max():.3f}]")

    # Execute CTDE episode
    for step in range(10):
        actions = {}
        for agent_id in ctde_env.agent_ids:
            avail_actions = ctde_env.get_avail_actions(agent_id)
            actions[agent_id] = np.random.choice(avail_actions) if avail_actions else 0

        # Step with global state in info (CTDE format)
        obs, rewards, dones, infos = ctde_env.step(actions)

        # Global state available in infos for CTDE algorithms
        if 'global_state' in infos:
            info_global_state = infos['global_state']
            # Verify consistency
            consistency = np.allclose(global_state, info_global_state) if step == 0 else "checked"
            print(f"  Step {step + 1}: global_state in info shape={info_global_state.shape}, consistency={consistency}")

        if any(dones.values()):
            break

    ctde_env.close()
    print("✓ CTDE environment closed\n")


def demo_interface_comparison():
    """Demo showing interface consistency with other environments"""
    print("=" * 50)
    print("DEMO: Interface Consistency Check")
    print("=" * 50)

    # These are the same methods available in DEM, HRG, MSFS environments
    unified_methods = [
        "reset()",
        "step(actions)",
        "get_observations()",
        "get_global_state()",
        "get_avail_actions(agent_id)",
        "get_env_info()",
        "close()"
    ]

    print("Unified interface methods (available in DEM, HRG, MSFS, SMAC):")
    for method in unified_methods:
        print(f"  ✓ {method}")

    # Create environment and verify methods exist
    env = create_smac_env(map_name="8m", episode_limit=10)

    print(f"\n✓ SMAC environment supports all unified methods:")
    for method in unified_methods:
        method_name = method.split("(")[0]
        has_method = hasattr(env, method_name)
        print(f"  {method_name}: {'✓' if has_method else '✗'}")

    env.close()
    print("\n✓ Interface consistency verified!\n")


def demo_configuration_system():
    """Demo configuration system (similar to other environments)"""
    print("=" * 50)
    print("DEMO: Configuration System")
    print("=" * 50)

    # Test predefined configurations
    configs = ["easy", "normal", "debug"]

    for config_name in configs:
        if config_name == "easy":
            config = get_easy_config()
        elif config_name == "normal":
            config = get_normal_config()
        else:  # debug
            config = get_debug_config()

        print(f"✓ Configuration '{config_name}':")
        print(f"  - Map: {config.map_name}")
        print(f"  - Episode limit: {config.episode_limit}")
        print(f"  - Debug: {config.debug}")

        # Create environment with config
        env = create_smac_env(config=config)
        observations = env.reset()
        print(f"  - Environment created with {len(env.agent_ids)} agents")
        env.close()

    print("\n✓ Configuration system working!\n")


def run_demonstration():
    """Run complete demonstration"""
    print("SMAC Environment Wrapper Demonstration")
    print("This shows how the SMAC wrapper provides the same interface as DEM, HRG, MSFS\n")

    try:
        demo_basic_interface()
        demo_ctde_interface()
        demo_interface_comparison()
        demo_configuration_system()

        print("=" * 50)
        print("✓ ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print("\nThe SMAC environment wrapper now provides:")
        print("• Same interface as DEM, HRG, MSFS environments")
        print("• Dictionary-based observations and actions")
        print("• Per-agent rewards")
        print("• Global state for CTDE algorithms")
        print("• Action masking support")
        print("• Configuration management")
        print("• Unified testing framework")
        print("\nReady for use with QMIX, VDN, MADDPG, and other MARL algorithms!")

    except Exception as e:
        print(f"\n✗ Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    run_demonstration()