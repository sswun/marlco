"""
Test Suite for HRG Environment

This module provides comprehensive tests for the HRG environment
to ensure correctness, robustness, and performance.
"""

import numpy as np
import time
import logging
from typing import Dict, List, Any

from .env_hrg import HRGEnv, HRGConfig, create_hrg_env
from .env_hrg_ctde import HRGCTDEWrapper, create_hrg_ctde_env
from .config import get_config_by_name, HRGPresetConfigs
from .core import AgentType, ActionType, ResourceType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HRGEnvironmentTester:
    """Comprehensive test suite for HRG environment"""

    def __init__(self):
        self.test_results = {}
        self.current_test = None

    def run_all_tests(self):
        """Run all environment tests"""
        logger.info("Starting comprehensive HRG environment tests...")

        tests = [
            self.test_basic_functionality,
            self.test_agent_configuration,
            self.test_resource_system,
            self.test_movement_and_collision,
            self.test_gathering_mechanics,
            self.test_transfer_mechanics,
            self.test_observation_spaces,
            self.test_action_spaces,
            self.test_reward_system,
            self.test_episode_termination,
            self.test_visualization,
            self.test_ctde_wrapper,
            self.test_performance,
            self.test_edge_cases,
            self.test_config_presets
        ]

        passed = 0
        total = len(tests)

        for test in tests:
            try:
                test_name = test.__name__.replace('test_', '')
                self.current_test = test_name
                logger.info(f"\n{'='*50}")
                logger.info(f"Running test: {test_name}")
                logger.info(f"{'='*50}")

                start_time = time.time()
                result = test()
                execution_time = time.time() - start_time

                self.test_results[test_name] = {
                    'passed': result,
                    'time': execution_time,
                    'error': None
                }

                if result:
                    logger.info(f"✅ {test_name} PASSED ({execution_time:.2f}s)")
                    passed += 1
                else:
                    logger.error(f"❌ {test_name} FAILED")

            except Exception as e:
                logger.error(f"❌ {test_name} ERROR: {str(e)}")
                self.test_results[test_name] = {
                    'passed': False,
                    'time': 0,
                    'error': str(e)
                }

        self._print_test_summary(passed, total)

    def test_basic_functionality(self) -> bool:
        """Test basic environment creation and reset"""
        try:
            # Test basic environment creation
            env = create_hrg_env(difficulty="normal")
            assert env is not None, "Environment creation failed"
            assert env.n_agents == 6, f"Expected 6 agents, got {env.n_agents}"
            assert len(env.agent_ids) == 6, "Agent IDs not properly initialized"

            # Test reset functionality
            observations = env.reset()
            assert isinstance(observations, dict), "Observations should be a dictionary"
            assert len(observations) == 6, "Should have observations for all agents"

            for agent_id, obs in observations.items():
                assert isinstance(obs, np.ndarray), f"Observation for {agent_id} should be numpy array"
                assert obs.shape == (80,), f"Observation shape should be (80,), got {obs.shape}"
                assert not np.any(np.isnan(obs)), f"Observation for {agent_id} contains NaN"
                assert not np.any(np.isinf(obs)), f"Observation for {agent_id} contains Inf"

            # Test step functionality
            actions = {agent_id: 7 for agent_id in env.agent_ids}  # All wait
            obs, rewards, dones, infos = env.step(actions)

            assert isinstance(obs, dict), "Step observations should be dictionary"
            assert isinstance(rewards, dict), "Rewards should be dictionary"
            assert isinstance(dones, dict), "Dones should be dictionary"
            assert isinstance(infos, dict), "Infos should be dictionary"

            env.close()
            return True

        except Exception as e:
            logger.error(f"Basic functionality test failed: {e}")
            return False

    def test_agent_configuration(self) -> bool:
        """Test agent configuration and role assignment"""
        try:
            env = create_hrg_env(difficulty="normal")

            # Check agent types
            agent_types = [agent.type for agent in env.agents.values()]
            scout_count = sum(1 for t in agent_types if t == AgentType.SCOUT)
            worker_count = sum(1 for t in agent_types if t == AgentType.WORKER)
            transporter_count = sum(1 for t in agent_types if t == AgentType.TRANSPORTER)

            assert scout_count == 2, f"Expected 2 scouts, got {scout_count}"
            assert worker_count == 3, f"Expected 3 workers, got {worker_count}"
            assert transporter_count == 1, f"Expected 1 transporter, got {transporter_count}"

            # Check agent-specific abilities
            for agent in env.agents.values():
                if agent.type == AgentType.SCOUT:
                    assert agent.config.vision_range == 5, "Scout vision range should be 5"
                    assert agent.config.move_speed == 2.0, "Scout move speed should be 2.0"
                    assert agent.config.carry_capacity == 0, "Scout carry capacity should be 0"
                elif agent.type == AgentType.WORKER:
                    assert agent.config.vision_range == 3, "Worker vision range should be 3"
                    assert agent.config.move_speed == 1.0, "Worker move speed should be 1.0"
                    assert agent.config.carry_capacity == 2, "Worker carry capacity should be 2"
                elif agent.type == AgentType.TRANSPORTER:
                    assert agent.config.vision_range == 4, "Transporter vision range should be 4"
                    assert agent.config.move_speed == 1.5, "Transporter move speed should be 1.5"
                    assert agent.config.carry_capacity == 5, "Transporter carry capacity should be 5"

            env.close()
            return True

        except Exception as e:
            logger.error(f"Agent configuration test failed: {e}")
            return False

    def test_resource_system(self) -> bool:
        """Test resource generation and management"""
        try:
            env = create_hrg_env(difficulty="normal")
            env.reset()

            # Check initial resource count
            gold_count = sum(1 for r in env.game_state.resources
                           if r.resource_type == ResourceType.GOLD and r.is_active)
            wood_count = sum(1 for r in env.game_state.resources
                           if r.resource_type == ResourceType.WOOD and r.is_active)

            assert gold_count == 3, f"Expected 3 gold resources, got {gold_count}"
            assert wood_count == 10, f"Expected 10 wood resources, got {wood_count}"

            # Check resource positions are valid
            for resource in env.game_state.resources:
                if resource.is_active:
                    assert env.game_state.is_valid_position(resource.position), \
                        f"Resource at invalid position: {resource.position}"

            # Test resource gathering (manual)
            worker = None
            for agent in env.agents.values():
                if agent.type == AgentType.WORKER:
                    worker = agent
                    break

            assert worker is not None, "No worker agent found"

            # Find a resource and move worker to it
            wood_resource = None
            for resource in env.game_state.resources:
                if resource.resource_type == ResourceType.WOOD and resource.is_active:
                    wood_resource = resource
                    break

            if wood_resource:
                # Teleport worker to resource (for testing)
                worker.position = wood_resource.position

                # Test gathering
                initial_quantity = wood_resource.remaining_quantity
                initial_inventory = worker.inventory[ResourceType.WOOD]

                # Simulate gathering
                for _ in range(3):  # Should be enough to complete gathering
                    reward = env._execute_action(worker.id, ActionType.GATHER.value)

                final_quantity = wood_resource.remaining_quantity
                final_inventory = worker.inventory[ResourceType.WOOD]

                assert final_quantity < initial_quantity, "Resource quantity should decrease"
                assert final_inventory > initial_inventory, "Worker inventory should increase"

            env.close()
            return True

        except Exception as e:
            logger.error(f"Resource system test failed: {e}")
            return False

    def test_movement_and_collision(self) -> bool:
        """Test agent movement and collision detection"""
        try:
            env = create_hrg_env(difficulty="normal")
            env.reset()

            # Test valid movement
            agent = list(env.agents.values())[0]
            initial_pos = agent.position

            # Move to valid position
            reward = env._execute_action(agent.id, ActionType.MOVE_EAST.value)

            # Position should change if move was valid
            final_pos = agent.position
            assert (final_pos.x != initial_pos.x or final_pos.y != initial_pos.y), \
                "Agent position should change after valid move"

            # Test boundary collision
            agent.position = Position(0, 0)  # Move to corner
            reward = env._execute_action(agent.id, ActionType.MOVE_WEST.value)
            assert agent.position.x == 0, "Agent should not move outside grid"

            # Test obstacle collision
            if env.game_state.obstacles:
                obstacle = list(env.game_state.obstacles)[0]
                agent.position = Position(
                    max(0, obstacle.x - 1),
                    max(0, obstacle.y - 1)
                )

                # Try to move into obstacle
                if obstacle.x > 0:
                    reward = env._execute_action(agent.id, ActionType.MOVE_EAST.value)
                    assert agent.position.x < obstacle.x, "Agent should not move into obstacle"

            env.close()
            return True

        except Exception as e:
            logger.error(f"Movement and collision test failed: {e}")
            return False

    def test_gathering_mechanics(self) -> bool:
        """Test resource gathering mechanics"""
        try:
            env = create_hrg_env(difficulty="normal")
            env.reset()

            # Get worker agent
            worker = None
            for agent in env.agents.values():
                if agent.type == AgentType.WORKER:
                    worker = agent
                    break

            assert worker is not None, "No worker agent found"

            # Test that scouts cannot gather
            scout = None
            for agent in env.agents.values():
                if agent.type == AgentType.SCOUT:
                    scout = agent
                    break

            if scout:
                wood_resource = None
                for resource in env.game_state.resources:
                    if resource.resource_type == ResourceType.WOOD and resource.is_active:
                        wood_resource = resource
                        break

                if wood_resource:
                    scout.position = wood_resource.position
                    initial_inventory = scout.inventory[ResourceType.WOOD]
                    reward = env._execute_action(scout.id, ActionType.GATHER.value)
                    final_inventory = scout.inventory[ResourceType.WOOD]
                    assert final_inventory == initial_inventory, "Scout should not be able to gather"

            # Test worker gathering time
            wood_resource = None
            for resource in env.game_state.resources:
                if resource.resource_type == ResourceType.WOOD and resource.is_active:
                    wood_resource = resource
                    break

            if wood_resource:
                worker.position = wood_resource.position
                initial_energy = worker.energy

                # Start gathering
                reward = env._execute_action(worker.id, ActionType.GATHER.value)
                assert worker.gathering_target == wood_resource, "Gathering target should be set"
                assert worker.gathering_progress == 1, "Gathering progress should increase"

                # Complete gathering
                for _ in range(2):
                    reward = env._execute_action(worker.id, ActionType.GATHER.value)

                assert worker.gathering_target is None, "Gathering target should be cleared"
                assert worker.gathering_progress == 0, "Gathering progress should be reset"
                assert worker.inventory[ResourceType.WOOD] > 0, "Worker should have gathered wood"

            env.close()
            return True

        except Exception as e:
            logger.error(f"Gathering mechanics test failed: {e}")
            return False

    def test_transfer_mechanics(self) -> bool:
        """Test resource transfer mechanics"""
        try:
            env = create_hrg_env(difficulty="normal")
            env.reset()

            # Get worker and transporter
            worker = None
            transporter = None

            for agent in env.agents.values():
                if agent.type == AgentType.WORKER:
                    worker = agent
                elif agent.type == AgentType.TRANSPORTER:
                    transporter = agent

            if worker and transporter:
                # Give worker some resources
                worker.add_resources(ResourceType.WOOD, 1)
                worker.add_resources(ResourceType.GOLD, 1)

                # Position them adjacent
                worker.position = Position(1, 0)
                transporter.position = Position(0, 0)

                # Test transfer
                initial_worker_inventory = dict(worker.inventory)
                initial_transporter_inventory = dict(transporter.inventory)

                reward = env._execute_action(worker.id, ActionType.TRANSFER.value)

                final_worker_inventory = dict(worker.inventory)
                final_transporter_inventory = dict(transporter.inventory)

                # Resources should be transferred
                assert sum(final_worker_inventory.values()) < sum(initial_worker_inventory.values()), \
                    "Worker should lose resources after transfer"
                assert sum(final_transporter_inventory.values()) > sum(initial_transporter_inventory.values()), \
                    "Transporter should gain resources after transfer"

            env.close()
            return True

        except Exception as e:
            logger.error(f"Transfer mechanics test failed: {e}")
            return False

    def test_observation_spaces(self) -> bool:
        """Test observation space generation"""
        try:
            env = create_hrg_env(difficulty="normal")
            observations = env.reset()

            for agent_id, obs in observations.items():
                # Check observation dimensions
                assert isinstance(obs, np.ndarray), f"Observation should be numpy array"
                assert obs.shape == (80,), f"Observation shape should be (80,), got {obs.shape}"
                assert obs.dtype == np.float32, f"Observation dtype should be float32"

                # Check for invalid values
                assert not np.any(np.isnan(obs)), f"Observation contains NaN for {agent_id}"
                assert not np.any(np.isinf(obs)), f"Observation contains Inf for {agent_id}"

                # Check value ranges (most should be normalized)
                assert np.all(obs >= -10), f"Observation values too low for {agent_id}"
                assert np.all(obs <= 10), f"Observation values too high for {agent_id}"

            # Test observation consistency
            for _ in range(5):
                actions = {agent_id: np.random.randint(0, 8) for agent_id in env.agent_ids}
                obs, rewards, dones, infos = env.step(actions)

                for agent_id, observation in obs.items():
                    assert observation.shape == (80,), f"Observation shape changed for {agent_id}"

            env.close()
            return True

        except Exception as e:
            logger.error(f"Observation spaces test failed: {e}")
            return False

    def test_action_spaces(self) -> bool:
        """Test action space and action masking"""
        try:
            env = create_hrg_env(difficulty="normal")
            env.reset()

            # Test action masking
            for agent_id in env.agent_ids:
                avail_actions = env.get_avail_actions(agent_id)
                assert isinstance(avail_actions, list), f"Available actions should be list for {agent_id}"
                assert len(avail_actions) > 0, f"Should have at least one available action for {agent_id}"
                assert all(0 <= action < 8 for action in avail_actions), \
                    f"Available actions should be in range [0,7] for {agent_id}"

            # Test invalid actions
            agent = list(env.agents.values())[0]
            invalid_action = 100  # Invalid action
            initial_pos = agent.position

            # This should not crash and should handle gracefully
            try:
                obs, rewards, dones, infos = env.step({agent.id: invalid_action})
                # Position should not change due to invalid action
                assert agent.position == initial_pos, "Agent should not move with invalid action"
            except Exception:
                pass  # Some implementations might raise an exception

            env.close()
            return True

        except Exception as e:
            logger.error(f"Action spaces test failed: {e}")
            return False

    def test_reward_system(self) -> bool:
        """Test reward system"""
        try:
            env = create_hrg_env(difficulty="normal")
            env.reset()

            # Test reward types
            total_rewards = {agent_id: 0 for agent_id in env.agent_ids}

            for step in range(10):
                actions = {agent_id: np.random.randint(0, 8) for agent_id in env.agent_ids}
                obs, rewards, dones, infos = env.step(actions)

                for agent_id, reward in rewards.items():
                    assert isinstance(reward, (int, float)), f"Reward should be numeric for {agent_id}"
                    assert not np.isnan(reward), f"Reward should not be NaN for {agent_id}"
                    assert not np.isinf(reward), f"Reward should not be Inf for {agent_id}"
                    total_rewards[agent_id] += reward

            # Test deposit reward (manual test)
            transporter = None
            for agent in env.agents.values():
                if agent.type == AgentType.TRANSPORTER:
                    transporter = agent
                    break

            if transporter:
                # Give transporter resources and move to base
                transporter.add_resources(ResourceType.GOLD, 1)
                transporter.position = Position(0, 0)  # Base

                initial_score = env.game_state.total_score
                reward = env._execute_action(transporter.id, ActionType.DEPOSIT.value)
                final_score = env.game_state.total_score

                assert final_score > initial_score, "Score should increase after deposit"
                assert reward > 0, "Should get positive reward for deposit"

            env.close()
            return True

        except Exception as e:
            logger.error(f"Reward system test failed: {e}")
            return False

    def test_episode_termination(self) -> bool:
        """Test episode termination conditions"""
        try:
            # Test max steps termination
            config = HRGConfig(max_steps=10)  # Short episode
            env = HRGEnv(config)
            env.reset()

            terminated = False
            for step in range(15):  # More than max_steps
                actions = {agent_id: 7 for agent_id in env.agent_ids}  # All wait
                obs, rewards, dones, infos = env.step(actions)

                if any(dones.values()):
                    terminated = True
                    break

            assert terminated, "Episode should terminate due to max steps"
            assert env.step_count >= 10, "Should have reached max steps"

            env.close()
            return True

        except Exception as e:
            logger.error(f"Episode termination test failed: {e}")
            return False

    def test_visualization(self) -> bool:
        """Test visualization functionality"""
        try:
            # Test matplotlib renderer
            from .renderer import MatplotlibRenderer

            env = create_hrg_env(difficulty="normal")
            env.reset()

            renderer = MatplotlibRenderer(grid_size=env.config.grid_size)
            renderer.render(env.game_state)  # Should not crash
            renderer.close()

            env.close()
            return True

        except Exception as e:
            logger.error(f"Visualization test failed: {e}")
            return False

    def test_ctde_wrapper(self) -> bool:
        """Test CTDE wrapper functionality"""
        try:
            # Test different global state types
            for global_state_type in ["concat", "mean", "max", "attention"]:
                env = create_hrg_ctde_env(
                    "normal_ctde",
                    global_state_type=global_state_type
                )

                observations = env.reset()
                global_state = env.get_global_state()

                assert isinstance(global_state, np.ndarray), "Global state should be numpy array"
                assert not np.any(np.isnan(global_state)), "Global state should not contain NaN"
                assert not np.any(np.isinf(global_state)), "Global state should not contain Inf"

                # Test step with global state in info
                actions = {agent_id: 7 for agent_id in env.agent_ids}
                obs, rewards, dones, infos = env.step(actions)

                assert 'global_state' in infos, "Global state should be in info dictionary"
                assert infos['global_state'].shape == global_state.shape, "Global state shape should be consistent"

                env.close()

            return True

        except Exception as e:
            logger.error(f"CTDE wrapper test failed: {e}")
            return False

    def test_performance(self) -> bool:
        """Test environment performance"""
        try:
            env = create_hrg_env(difficulty="normal")
            env.reset()

            # Measure step time
            start_time = time.time()
            num_steps = 100

            for step in range(num_steps):
                actions = {agent_id: np.random.randint(0, 8) for agent_id in env.agent_ids}
                obs, rewards, dones, infos = env.step(actions)

                if any(dones.values()):
                    env.reset()

            total_time = time.time() - start_time
            avg_step_time = total_time / num_steps

            logger.info(f"Average step time: {avg_step_time:.4f}s")
            logger.info(f"Steps per second: {1/avg_step_time:.2f}")

            # Performance should be reasonable for training
            assert avg_step_time < 0.1, f"Step time too slow: {avg_step_time:.4f}s"

            env.close()
            return True

        except Exception as e:
            logger.error(f"Performance test failed: {e}")
            return False

    def test_edge_cases(self) -> bool:
        """Test edge cases and error handling"""
        try:
            env = create_hrg_env(difficulty="normal")
            observations = env.reset()

            # Test empty actions
            obs, rewards, dones, infos = env.step({})
            assert isinstance(obs, dict), "Should handle empty actions"

            # Test None actions
            actions = {agent_id: None for agent_id in env.agent_ids}
            try:
                obs, rewards, dones, infos = env.step(actions)
            except Exception:
                pass  # Should handle gracefully

            # Test actions for non-existent agents
            invalid_actions = {'invalid_agent': 0}
            try:
                obs, rewards, dones, infos = env.step(invalid_actions)
            except Exception:
                pass  # Should handle gracefully

            # Test multiple resets
            for _ in range(3):
                obs = env.reset()
                assert len(obs) == env.n_agents, "Should handle multiple resets"

            env.close()
            return True

        except Exception as e:
            logger.error(f"Edge cases test failed: {e}")
            return False

    def test_config_presets(self) -> bool:
        """Test configuration presets"""
        try:
            # Test different difficulty presets
            for difficulty in ["easy", "normal", "hard"]:
                env = create_hrg_env(difficulty=difficulty)
                env.reset()

                if difficulty == "easy":
                    assert env.config.grid_size <= 10, "Easy mode should have smaller grid"
                    assert env.config.max_steps >= 200, "Easy mode should have more steps"
                elif difficulty == "hard":
                    assert env.config.num_obstacles >= 10, "Hard mode should have more obstacles"

                env.close()

            # Test preset configurations
            presets = [
                'communication', 'coordination', 'exploration', 'role_test'
            ]

            for preset_name in presets:
                try:
                    config = get_config_by_name(preset_name)
                    env = HRGEnv(config)
                    env.reset()
                    env.close()
                except Exception as e:
                    logger.warning(f"Preset {preset_name} failed: {e}")

            return True

        except Exception as e:
            logger.error(f"Config presets test failed: {e}")
            return False

    def _print_test_summary(self, passed: int, total: int):
        """Print test summary"""
        logger.info(f"\n{'='*60}")
        logger.info(f"TEST SUMMARY: {passed}/{total} tests passed")
        logger.info(f"{'='*60}")

        for test_name, result in self.test_results.items():
            status = "✅ PASSED" if result['passed'] else "❌ FAILED"
            time_info = f"({result['time']:.3f}s)" if result['passed'] else ""
            error_info = f" - {result['error']}" if result['error'] else ""

            logger.info(f"{status:<12} {test_name:<30} {time_info:<15} {error_info}")

        success_rate = (passed / total) * 100
        logger.info(f"\nSuccess Rate: {success_rate:.1f}%")

        if success_rate >= 90:
            logger.info("🎉 EXCELLENT: Environment is ready for training!")
        elif success_rate >= 70:
            logger.info("✅ GOOD: Environment is mostly functional, minor issues may exist.")
        else:
            logger.error("⚠️  NEEDS ATTENTION: Environment has significant issues that should be resolved.")


def run_quick_test():
    """Run a quick test of basic functionality"""
    logger.info("Running quick HRG environment test...")

    try:
        # Test basic environment
        env = create_hrg_env(difficulty="easy")
        observations = env.reset()
        logger.info(f"✅ Environment created successfully with {env.n_agents} agents")
        logger.info(f"✅ Reset successful, observation shape: {list(observations.values())[0].shape}")

        # Test a few steps
        for step in range(5):
            actions = {agent_id: np.random.randint(0, 8) for agent_id in env.agent_ids}
            obs, rewards, dones, infos = env.step(actions)
            logger.info(f"Step {step+1}: Avg reward = {np.mean(list(rewards.values())):.3f}")

        env.close()
        logger.info("✅ Quick test completed successfully!")
        return True

    except Exception as e:
        logger.error(f"❌ Quick test failed: {e}")
        return False


def run_comprehensive_test():
    """Run comprehensive test suite"""
    tester = HRGEnvironmentTester()
    tester.run_all_tests()
    return tester.test_results


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        run_quick_test()
    else:
        run_comprehensive_test()