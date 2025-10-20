"""
IQL 快速测试脚本 - 运行一个简单的训练示例
"""
import sys
import os
from src.trainer import IQLTrainer
from src.utils import load_config, set_seed


def main():
    """运行快速测试"""
    print("🚀 开始IQL快速测试...")

    # 设置随机种子
    set_seed(42)

    # 加载配置（使用较小的参数进行快速测试）
    try:
        config = load_config('config.yaml')

        # 修改为快速测试参数
        config['training']['total_episodes'] = 20
        config['training']['batch_size'] = 16
        config['training']['buffer_size'] = 50
        config['training']['warmup_episodes'] = 5
        config['training']['eval_interval'] = 10
        config['training']['save_interval'] = 100  # 不保存中间检查点

        # 使用简单的环境
        config['env']['name'] = 'CM'
        config['env']['difficulty'] = 'debug' if hasattr(config['env'], 'debug') else 'easy'

        print(f"✅ 配置已加载并修改为快速测试模式")

    except FileNotFoundError:
        print("❌ 配置文件不存在，使用默认配置")
        # 创建默认配置
        config = {
            'env': {
                'name': 'CM',
                'difficulty': 'easy_ctde',
                'global_state_type': 'concat'
            },
            'algorithm': {
                'gamma': 0.99,
                'learning_rate': 0.001,
                'tau': 0.005,
                'target_update_interval': 10,
                'max_grad_norm': 10.0
            },
            'model': {
                'hidden_dim': 64
            },
            'training': {
                'total_episodes': 20,
                'batch_size': 16,
                'buffer_size': 50,
                'warmup_episodes': 5,
                'eval_interval': 10,
                'save_interval': 100
            },
            'exploration': {
                'epsilon_start': 0.5,
                'epsilon_end': 0.1,
                'epsilon_decay': 0.95
            }
        }

    print(f"🎮 环境: {config['env']['name']}")
    print(f"🎯 难度: {config['env']['difficulty']}")
    print(f"🔄 Episodes: {config['training']['total_episodes']}")

    # 创建训练器
    try:
        trainer = IQLTrainer(config)
        print("✅ 训练器创建成功")
    except Exception as e:
        print(f"❌ 训练器创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 开始训练
    try:
        print("\n🏃‍♂️ 开始训练...")
        episode_rewards = trainer.train()

        # 打印结果
        if episode_rewards:
            avg_reward = sum(episode_rewards[-5:]) / min(5, len(episode_rewards))
            print(f"\n📊 训练结果:")
            print(f"   最终平均奖励 (last 5): {avg_reward:.2f}")
            print(f"   总episodes: {len(episode_rewards)}")
            print(f"   最佳奖励: {max(episode_rewards):.2f}")

            # 最终评估
            final_eval = trainer.evaluate(num_episodes=3)
            print(f"   最终评估分数: {final_eval:.2f}")

        print("\n✅ 快速测试完成！IQL算法运行正常。")
        return True

    except Exception as e:
        print(f"\n❌ 训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        trainer.close()
        print("🔚 训练器已关闭")


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)