"""
IQL 主训练脚本
"""
import argparse
import os
import sys
from src.trainer import IQLTrainer
from src.utils import load_config, set_seed, save_training_plots


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='IQL Multi-Agent RL Training')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='配置文件路径')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--env', type=str, choices=['DEM', 'HRG', 'MSFS', 'CM', 'SMAC', 'multiwalker', 'simple_spread', 'simple_crypto'],
                       help='环境名称 (覆盖配置文件)')
    parser.add_argument('--difficulty', type=str,
                       help='难度级别/地图名称 (覆盖配置文件)\n'
                            '对于DEM/HRG/MSFS/CM: easy, normal, hard\n'
                            '对于SMAC: 地图名称 (8m, 3s5z, MMM等) 或 easy/normal/hard\n'
                            '对于PettingZoo: default')
    parser.add_argument('--episodes', type=int,
                       help='训练episodes数 (覆盖配置文件)')
    parser.add_argument('--plots', action='store_true',
                       help='Generate and save training plots')
    parser.add_argument('--show-plots', action='store_true',
                       help='Display plots after training')
    parser.add_argument('--plot-dir', type=str, default='plots',
                       help='Directory to save plots')

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 加载配置
    try:
        config = load_config(args.config)
        print(f"✅ 配置已加载: {args.config}")
    except FileNotFoundError:
        print(f"❌ 配置文件不存在: {args.config}")
        return
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        return

    # 命令行参数覆盖配置
    if args.env:
        config['env']['name'] = args.env
    if args.difficulty:
        config['env']['difficulty'] = args.difficulty
    if args.episodes:
        config['training']['total_episodes'] = args.episodes

    print(f"🎮 环境: {config['env']['name']}")
    print(f"🎯 难度: {config['env']['difficulty']}")
    print(f"🔄 Episodes: {config['training']['total_episodes']}")
    print(f"🎲 随机种子: {args.seed}")

    # 创建checkpoints目录
    os.makedirs('checkpoints', exist_ok=True)

    # 创建plots目录 if needed
    if args.plots or args.show_plots:
        os.makedirs(args.plot_dir, exist_ok=True)
        print(f"📊 Plots will be saved to: {args.plot_dir}")

    # 创建训练器
    try:
        trainer = IQLTrainer(config)
        print("✅ 训练器创建成功")
    except Exception as e:
        print(f"❌ 训练器创建失败: {e}")
        return

    # 开始训练
    try:
        episode_rewards = trainer.train()

        # 打印训练结果统计
        import numpy as np
        avg_reward = np.mean(episode_rewards[-1000:])  # 最后1000个episode的平均奖励
        print(f"\n📊 训练结果统计:")
        print(f"   最终平均奖励: {avg_reward:.2f}")
        print(f"   总episodes: {len(episode_rewards)}")

        # 最终评估
        final_eval = trainer.evaluate(num_episodes=20)
        print(f"   最终评估分数: {final_eval:.2f}")

        # 保存训练数据
        print(f"\n💾 保存训练数据...")
        try:
            data_file = trainer.save_training_data(save_dir='checkpoints')
            print(f"✅ 数据文件已保存")
        except Exception as e:
            print(f"⚠️  保存数据失败: {e}")

        # 生成图表
        if args.plots or args.show_plots:
            print("\n📊 Generating training plots...")
            try:
                metrics = trainer.get_training_metrics()
                env_name = f"{config['env']['name']}_{config['env']['difficulty']}"

                saved_files = save_training_plots(
                    episode_rewards=metrics['episode_rewards'],
                    episode_lengths=metrics['episode_lengths'],
                    losses=metrics['losses'],
                    eval_episodes=metrics['eval_episodes'],
                    eval_rewards=metrics['eval_rewards'],
                    epsilon_values=metrics['epsilon_history'],
                    save_dir=args.plot_dir,
                    env_name=env_name,
                    config=config,
                    show_plots=args.show_plots
                )

                print(f"✅ Successfully generated {len(saved_files)} plots:")
                for file_path in saved_files:
                    print(f"   📈 {file_path}")

            except Exception as e:
                print(f"❌ Error generating plots: {e}")
                import traceback
                traceback.print_exc()

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")

        # 即使中断也尝试生成图表
        if args.plots or args.show_plots:
            print("\n📊 Generating plots for interrupted training...")
            try:
                metrics = trainer.get_training_metrics()
                if len(metrics['episode_rewards']) > 0:
                    env_name = f"{config['env']['name']}_{config['env']['difficulty']}"

                    saved_files = save_training_plots(
                        episode_rewards=metrics['episode_rewards'],
                        episode_lengths=metrics['episode_lengths'],
                        losses=metrics['losses'],
                        eval_episodes=metrics['eval_episodes'],
                        eval_rewards=metrics['eval_rewards'],
                        epsilon_values=metrics['epsilon_history'],
                        save_dir=args.plot_dir,
                        env_name=env_name,
                        config=config,
                        show_plots=args.show_plots
                    )

                    print(f"✅ Generated {len(saved_files)} plots from interrupted training:")
                    for file_path in saved_files:
                        print(f"   📈 {file_path}")
            except Exception as e:
                print(f"❌ Could not generate plots: {e}")

    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        trainer.close()
        print("🔚 Trainer closed")


if __name__ == "__main__":
    main()