#!/usr/bin/env python3
"""
Test the step format fix for CM environment
"""

import sys
import os

# Add project root to path
sys.path.insert(0, '/mnt/j/files/A_projects_extension/code_python_mulitiagent/MARL')

def test_cm_environment_step_format():
    """Test that CM environment step() returns correct format"""
    print("Testing CM environment step format...")

    try:
        from Env.CM import create_cm_env, create_cm_ctde_env

        # Test basic environment
        env = create_cm_env(difficulty="debug", render_mode=None)
        obs, info = env.reset()

        actions = {agent_id: env.get_avail_actions(agent_id)[0] for agent_id in env.agent_ids}
        result = env.step(actions)

        print(f"Basic environment step result length: {len(result)}")

        if len(result) == 4:
            obs, rewards, done, info = result
            print("✅ Basic environment returns 4 values correctly")
            env.close()
        else:
            print(f"❌ Basic environment returns {len(result)} values, expected 4")
            env.close()
            return False

        # Test CTDE environment
        ctde_env = create_cm_ctde_env(difficulty="debug", global_state_type="concat")
        obs, global_state = ctde_env.reset()

        actions = {agent_id: 0 for agent_id in ctde_env.agent_ids}
        result = ctde_env.step(actions)

        print(f"CTDE environment step result length: {len(result)}")

        if len(result) == 5:
            obs, rewards, terminated, truncated, info = result
            print("✅ CTDE environment returns 5 values correctly")
            print(f"  - terminated: {terminated}")
            print(f"  - truncated: {truncated}")
            print(f"  - global_state in info: {'global_state' in info}")
            ctde_env.close()
            return True
        else:
            print(f"❌ CTDE environment returns {len(result)} values, expected 5")
            ctde_env.close()
            return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_marlv2_trainer_compatibility():
    """Test compatibility with marlv2 trainer"""
    print("\nTesting marlv2 trainer compatibility...")

    try:
        from Env.CM import create_cm_ctde_env
        from marlv2.trainer import QMIXTrainer
        from marlv2.config.qmix_config import get_debug_config

        # Create environment
        env = create_cm_ctde_env(difficulty="debug", global_state_type="concat")

        # Create minimal config
        config = get_debug_config()
        config.experiment_name = "test_compatibility"
        config.environment.max_episodes = 1  # Just test one episode

        # Create logger
        from marlv2.utils.logger import Logger
        logger = Logger(
            experiment_name="test",
            log_dir="test_logs",
            use_tensorboard=False,
            log_to_console=False,
            log_to_file=False
        )

        # Try to create trainer
        trainer = QMIXTrainer(env=env, config=config, logger=logger)
        print("✅ QMIXTrainer created successfully")

        # Try to run one episode
        try:
            episode_result = trainer._run_episode()
            print("✅ One episode completed successfully")
            print(f"  Episode reward: {episode_result['episode_reward']:.3f}")
            print(f"  Episode length: {episode_result['episode_length']}")

            trainer.env.close()
            logger.close()
            return True

        except Exception as e:
            print(f"❌ Episode execution failed: {e}")
            trainer.env.close()
            logger.close()
            return False

    except Exception as e:
        print(f"❌ Trainer compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("CM Environment Step Format Test")
    print("=" * 60)

    test1_passed = test_cm_environment_step_format()
    test2_passed = test_marlv2_trainer_compatibility()

    print("\n" + "=" * 60)
    print(f"Test Results:")
    print(f"  Step format test: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"  Marlv2 compatibility test: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    print("=" * 60)

    if test1_passed and test2_passed:
        print("🎉 All tests passed! Step format fix is working correctly.")
        print("The CM environment should now work with marlv2 trainer.")
    else:
        print("⚠️ Some tests failed. Step format fix needs more work.")