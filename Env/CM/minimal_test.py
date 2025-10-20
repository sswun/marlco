#!/usr/bin/env python3
"""
Minimal test for CM environment core functionality
"""

import sys
import os

# Simple test without complex dependencies
def test_imports():
    """Test if we can import the basic modules"""
    print("Testing imports...")

    try:
        sys.path.insert(0, '/mnt/j/files/A_projects_extension/code_python_mulitiagent/MARL')
        from Env.CM.core import Position, Box, Goal, Agent, CMGameState, ActionType
        print("✅ Core classes imported successfully")

        from Env.CM.config import CMConfig, get_easy_config
        print("✅ Configuration imported successfully")

        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_core_functionality():
    """Test core classes without environment"""
    print("\nTesting core functionality...")

    try:
        from Env.CM.core import Position, Box, Goal, Agent, CMGameState, ActionType
        from Env.CM.config import get_easy_config

        # Create basic objects
        pos = Position(2, 3)
        box = Box(pos, 2)
        goal = Goal(Position(5, 5), 2)
        agent = Agent("agent_0", Position(2, 1))

        print(f"✅ Created basic objects: pos={pos}, box_size={box.size}")

        # Test game state
        agents = [agent]
        game_state = CMGameState(agents, box, goal, 7)

        print(f"✅ Created game state with {len(game_state.agents)} agents")
        print(f"   Box position: {game_state.box.position}")
        print(f"   Goal position: {game_state.goal.position}")
        print(f"   Distance to goal: {game_state.box.get_center()[0] - game_state.goal.get_center()[0]:.1f}")

        # Test agent movement
        old_pos = agent.position.copy()
        success = agent.move(ActionType.MOVE_RIGHT, 7)
        new_pos = agent.position

        print(f"✅ Agent movement: {old_pos} -> {new_pos}, success={success}")

        # Test box pushing detection
        pushing = agent.is_pushing_box(box)
        print(f"✅ Pushing detection: {pushing}")

        return True
    except Exception as e:
        print(f"❌ Core functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_configuration():
    """Test configuration system"""
    print("\nTesting configuration...")

    try:
        from Env.CM.config import CMConfig, get_easy_config, list_available_configs

        # Test basic config
        config = CMConfig(grid_size=5, n_agents=2)
        print(f"✅ Created basic config: {config.grid_size}x{config.grid_size}, {config.n_agents} agents")

        # Test predefined configs
        easy_config = get_easy_config()
        print(f"✅ Easy config: {easy_config.grid_size}x{easy_config.grid_size}, {easy_config.n_agents} agents")

        # Test config listing
        configs = list_available_configs()
        print(f"✅ Available configs: {len(configs)} configurations")

        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("CM Environment Minimal Test")
    print("=" * 50)

    success = True

    success &= test_imports()
    success &= test_core_functionality()
    success &= test_configuration()

    print("\n" + "=" * 50)
    if success:
        print("🎉 Minimal tests passed! Core functionality works.")
    else:
        print("❌ Some tests failed.")
    print("=" * 50)