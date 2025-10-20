"""
VDN 主训练脚本 - Value Decomposition Networks
"""
import argparse
import os
import sys
import time
import logging
from typing import Dict, Any, List

# 添加当前目录到路径
sys.path.append(os.path.dirname(__file__))

from src.trainer import VDNTrainer
from src.utils import set_seed, load_config, save_training_plots, validate_config, log_experiment_info


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='VDN Multi-Agent RL Training')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='配置文件路径')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--env', type=str,
                       choices=['DEM', 'HRG', 'MSFS', 'CM', 'SMAC', 'multiwalker', 'simple_spread', 'simple_crypto'],
                       help='环境名称 (覆盖配置文件)')
    parser.add_argument('--difficulty', type=str,
                       help='难度级别/地图名称 (覆盖配置文件)\n'
                            '对于DEM/HRG/MSFS/CM: easy, normal, hard\n'
                            '对于SMAC: 地图名称 (8m, 3s5z, MMM) 或 easy/normal/hard\n'
                            '对于PettingZoo: default')
    parser.add_argument('--episodes', type=int,
                       help='训练episodes数 (覆盖配置文件)')
    parser.add_argument('--plots', action='store_true',
                       help='生成并保存训练图表')
    parser.add_argument('--show-plots', action='store_true',
                       help='训练结束后显示图表')
    parser.add_argument('--plot-dir', type=str, default='plots',
                       help='图表保存目录')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='模型保存目录')
    parser.add_argument('--eval-episodes', type=int, default=20,
                       help='评估episodes数量')
    parser.add_argument('--save-interval', type=int,
                       help='模型保存间隔 (覆盖配置文件)')
    parser.add_argument('--eval-interval', type=int,
                       help='评估间隔 (覆盖配置文件)')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'auto'], default='auto',
                       help='计算设备 (auto: 自动选择)')

    return parser.parse_args()


def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('vdn_training.log', mode='w')
        ]
    )
    return logging.getLogger(__name__)


def validate_and_update_config(config: Dict[str, Any], args) -> Dict[str, Any]:
    """验证并更新配置"""
    # 验证配置
    if not validate_config(config):
        raise ValueError("配置文件验证失败")

    # 命令行参数覆盖配置
    if args.env:
        config['env']['name'] = args.env
    if args.difficulty:
        config['env']['difficulty'] = args.difficulty
    if args.episodes:
        config['training']['total_episodes'] = args.episodes
    if args.save_interval:
        config['training']['save_interval'] = args.save_interval
    if args.eval_interval:
        config['training']['eval_interval'] = args.eval_interval

    return config


def get_device(args):
    """获取计算设备"""
    if args.device == 'auto':
        from src.utils import get_device
        return get_device()
    else:
        import torch
        if args.device == 'cuda' and not torch.cuda.is_available():
            logging.warning("CUDA不可用，将使用CPU")
            return torch.device('cpu')
        return torch.device(args.device)


def create_directories(plot_dir: str, checkpoint_dir: str):
    """创建必要的目录"""
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    logging.info(f"创建目录: {plot_dir}, {checkpoint_dir}")


def print_training_info(config: Dict[str, Any], device, args):
    """打印训练信息"""
    print("\n" + "="*60)
    print("VDN (Value Decomposition Networks) 训练")
    print("="*60)
    print(f"🎮 环境: {config['env']['name']}")
    print(f"🎯 难度: {config['env']['difficulty']}")
    print(f"🔄 Episodes: {config['training']['total_episodes']}")
    print(f"🎲 随机种子: {args.seed}")
    print(f"💻 设备: {device}")
    print(f"📊 批次大小: {config['training']['batch_size']}")
    print(f"🧠 学习率: {config['algorithm']['learning_rate']}")
    print(f"📈 探索率: {config['exploration']['epsilon_start']} -> {config['exploration']['epsilon_end']}")
    print("="*60)


def main():
    """主函数"""
    # 解析参数
    args = parse_args()

    # 设置日志
    logger = setup_logging()

    try:
        # 设置随机种子
        set_seed(args.seed)

        # 加载配置
        try:
            config = load_config(args.config)
            logger.info(f"✅ 配置已加载: {args.config}")
        except FileNotFoundError:
            logger.error(f"❌ 配置文件不存在: {args.config}")
            return 1
        except Exception as e:
            logger.error(f"❌ 配置文件加载失败: {e}")
            return 1

        # 验证并更新配置
        config = validate_and_update_config(config, args)

        # 获取设备
        device = get_device(args)

        # 创建目录
        create_directories(args.plot_dir, args.checkpoint_dir)

        # 打印训练信息
        print_training_info(config, device, args)

        # 创建训练器
        try:
            trainer = VDNTrainer(config, device)
            logger.info("✅ VDN训练器创建成功")
        except Exception as e:
            logger.error(f"❌ 训练器创建失败: {e}")
            import traceback
            traceback.print_exc()
            return 1

        # 记录实验信息
        env_info = trainer.env.get_env_info()
        log_experiment_info(config, env_info)

        # 记录开始时间
        start_time = time.time()

        # 开始训练
        try:
            logger.info("🚀 开始训练...")
            episode_rewards = trainer.train()

            # 训练完成统计
            training_time = time.time() - start_time
            logger.info(f"\n🎉 训练完成！")
            logger.info(f"⏱️  训练时间: {training_time/3600:.2f} 小时")
            logger.info(f"📊 总episodes: {len(episode_rewards)}")

            # 打印训练结果统计
            import numpy as np
            if len(episode_rewards) >= 100:
                avg_reward = np.mean(episode_rewards[-100:])
                logger.info(f"📈 最后100个episode平均奖励: {avg_reward:.2f}")

            best_reward = max(episode_rewards)
            final_reward = episode_rewards[-1]
            logger.info(f"🏆 最佳奖励: {best_reward:.2f}")
            logger.info(f"🎯 最终奖励: {final_reward:.2f}")

            # 最终评估
            logger.info(f"\n🧪 执行最终评估 ({args.eval_episodes} episodes)...")
            final_eval = trainer.evaluate(num_episodes=args.eval_episodes)
            logger.info(f"📋 最终评估分数: {final_eval:.2f}")

            # 保存训练数据
            logger.info(f"\n💾 保存训练数据...")
            try:
                data_file = trainer.save_training_data(save_dir=args.checkpoint_dir)
                logger.info(f"✅ 数据文件已保存: {data_file}")
            except Exception as e:
                logger.warning(f"⚠️  保存数据失败: {e}")

            # 生成图表
            if args.plots or args.show_plots:
                logger.info(f"\n📊 生成训练图表...")
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

                    logger.info(f"✅ 成功生成 {len(saved_files)} 个图表:")
                    for file_path in saved_files:
                        logger.info(f"   📈 {file_path}")

                except Exception as e:
                    logger.error(f"❌ 生成图表时发生错误: {e}")
                    import traceback
                    traceback.print_exc()

            logger.info(f"\n🎊 VDN训练任务完成！")
            return 0

        except KeyboardInterrupt:
            logger.warning(f"\n⚠️  训练被用户中断")

            # 即使中断也尝试生成图表
            if args.plots or args.show_plots:
                logger.info(f"\n📊 为中断的训练生成图表...")
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

                        logger.info(f"✅ 从中断训练生成了 {len(saved_files)} 个图表:")
                        for file_path in saved_files:
                            logger.info(f"   📈 {file_path}")
                except Exception as e:
                    logger.error(f"❌ 生成图表失败: {e}")

            return 1

        except Exception as e:
            logger.error(f"\n❌ 训练过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            return 1

        finally:
            # 清理资源
            trainer.close()
            logger.info("🔚 训练器已关闭")

    except Exception as e:
        logger.error(f"❌ 程序执行失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)