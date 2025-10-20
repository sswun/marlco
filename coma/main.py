"""
COMA 主训练脚本
与QMIX main.py保持完全一致的接口和功能
"""
import argparse
import os
import sys
import numpy as np

# 添加src路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.trainer import COMATrainer
from src.utils import load_config, set_seed


def parse_args():
    """解析命令行参数 - 与QMIX保持一致"""
    parser = argparse.ArgumentParser(description='COMA Multi-Agent RL Training')
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
    """主函数 - 与QMIX保持一致的逻辑"""
    args = parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 加载配置
    try:
        config = load_config(args.config)
        print(f"✅ COMA配置已加载: {args.config}")
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
    print(f"🔧 算法: COMA (Counterfactual Multi-Agent Policy Gradients)")

    # 创建checkpoints目录
    os.makedirs('checkpoints', exist_ok=True)

    # 创建plots目录 if needed
    if args.plots or args.show_plots:
        os.makedirs(args.plot_dir, exist_ok=True)
        print(f"📊 Plots will be saved to: {args.plot_dir}")

    # 创建训练器
    try:
        trainer = COMATrainer(config)
        print("✅ COMA训练器创建成功")
    except Exception as e:
        print(f"❌ 训练器创建失败: {e}")
        return

    # 开始训练
    try:
        episode_rewards, eval_rewards = trainer.train()

        # 打印训练结果统计
        avg_reward = np.mean(episode_rewards[-1000:]) if len(episode_rewards) >= 1000 else np.mean(episode_rewards)
        print(f"\n📊 COMA训练结果统计:")
        print(f"   最终平均奖励: {avg_reward:.2f}")
        print(f"   总episodes: {len(episode_rewards)}")

        # 最终评估
        final_eval = trainer._evaluate(n_episodes=20)
        print(f"   最终评估分数: {final_eval:.2f}")

        # 保存训练数据
        print(f"\n💾 保存COMA训练数据...")
        try:
            trainer.save_training_data()
            print(f"✅ COMA数据文件已保存")
        except Exception as e:
            print(f"⚠️  保存数据失败: {e}")

        # 生成图表
        if args.plots or args.show_plots:
            print("\n📊 生成COMA训练图表...")
            try:
                metrics = {
                    'episode_rewards': episode_rewards,
                    'episode_lengths': trainer.episode_lengths,
                    'losses': trainer.losses,
                    'eval_episodes': trainer.eval_episodes,
                    'eval_rewards': trainer.eval_rewards,
                    'epsilon_history': trainer.epsilon_history
                }
                env_name = f"COMA_{config['env']['name']}_{config['env']['difficulty']}"

                # 使用COMA完整的绘图功能
                from src.utils import save_training_plots
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
                print(f"❌ 生成图表时出错: {e}")
                import traceback
                traceback.print_exc()

    except KeyboardInterrupt:
        print("\n⚠️ COMA训练被用户中断")

        # 即使中断也尝试生成图表
        if args.plots or args.show_plots:
            print("\n📊 为中断的训练生成图表...")
            try:
                if hasattr(trainer, 'episode_rewards') and len(trainer.episode_rewards) > 0:
                    from src.utils import save_training_plots
                    env_name = f"COMA_{config['env']['name']}_{config['env']['difficulty']}"

                    saved_files = save_training_plots(
                        episode_rewards=trainer.episode_rewards,
                        episode_lengths=getattr(trainer, 'episode_lengths', None),
                        losses=getattr(trainer, 'losses', None),
                        eval_episodes=getattr(trainer, 'eval_episodes', None),
                        eval_rewards=getattr(trainer, 'eval_rewards', None),
                        epsilon_values=getattr(trainer, 'epsilon_history', None),
                        save_dir=args.plot_dir,
                        env_name=env_name + "_interrupted",
                        config=config,
                        show_plots=args.show_plots
                    )

                    print(f"✅ 中断训练图表已保存 ({len(saved_files)} 个):")
                    for file_path in saved_files:
                        print(f"   📈 {file_path}")

            except Exception as e:
                print(f"❌ 生成中断训练图表失败: {e}")

    print("\n🎉 COMA训练完成!")


if __name__ == "__main__":
    main()