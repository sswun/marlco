"""
工具函数
与QMIX utils保持一致
"""
import torch
import numpy as np
import random
import yaml
import os
from typing import Dict, Any, Union, Optional, List
import logging
import matplotlib.pyplot as plt
import matplotlib.style as style
from datetime import datetime
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """
    设置随机种子，确保实验可复现性

    Args:
        seed: 随机种子值
    """
    if not isinstance(seed, int):
        raise TypeError(f"seed必须是整数类型，当前类型为: {type(seed)}")

    if seed < 0:
        raise ValueError(f"seed必须是非负整数，当前值为: {seed}")

    try:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # 增强确定性
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        logger.info(f"随机种子已设置为: {seed}")
    except Exception as e:
        logger.error(f"设置随机种子时发生错误: {str(e)}")
        raise


def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典
    """
    if not isinstance(config_path, str):
        raise TypeError(f"config_path必须是字符串类型，当前类型为: {type(config_path)}")

    if not config_path.strip():
        raise ValueError("config_path不能为空字符串")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    if not os.path.isfile(config_path):
        raise ValueError(f"路径不是文件: {config_path}")

    # 检查文件扩展名
    valid_extensions = ['.yaml', '.yml']
    _, ext = os.path.splitext(config_path)
    if ext.lower() not in valid_extensions:
        logger.warning(f"配置文件扩展名不是标准的YAML格式: {ext}")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        if config is None:
            logger.warning(f"配置文件为空: {config_path}")
            return {}

        if not isinstance(config, dict):
            raise ValueError(f"配置文件内容必须是字典格式，当前类型为: {type(config)}")

        logger.info(f"成功加载配置文件: {config_path}")
        return config

    except yaml.YAMLError as e:
        logger.error(f"YAML解析错误: {str(e)}")
        raise ValueError(f"配置文件格式错误: {str(e)}")
    except Exception as e:
        logger.error(f"加载配置文件时发生错误: {str(e)}")
        raise


def get_device():
    """
    获取设备

    Returns:
        torch.device对象
    """
    try:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"使用CUDA设备: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            logger.info("CUDA不可用，使用CPU")

        return device

    except Exception as e:
        logger.error(f"获取设备时发生错误: {str(e)}")
        logger.info("回退到CPU设备")
        return torch.device('cpu')


def save_training_plots(episode_rewards: List[float],
                       episode_lengths: List[int] = None,
                       losses: List[float] = None,
                       eval_episodes: List[int] = None,
                       eval_rewards: List[float] = None,
                       epsilon_values: List[float] = None,
                       save_dir: str = "plots",
                       env_name: str = "Environment",
                       config: Dict[str, Any] = None,
                       show_plots: bool = False):
    """
    Generate and save all training plots as separate figures - 与QMIX完全相同

    Args:
        episode_rewards: List of episode rewards
        episode_lengths: List of episode lengths
        losses: List of training losses
        eval_episodes: List of episode numbers where evaluation was performed
        eval_rewards: List of evaluation rewards
        epsilon_values: List of epsilon values
        save_dir: Directory to save plots
        env_name: Name of the environment
        config: Training configuration
        show_plots: Whether to display plots

    Returns:
        List of file paths to saved plots
    """
    all_saved_files = []

    try:
        # 1. Episode rewards plot
        if episode_rewards:
            plt.figure(figsize=(12, 8))
            plt.plot(episode_rewards, alpha=0.7, linewidth=1)

            # Moving average
            if len(episode_rewards) > 100:
                window_size = min(100, len(episode_rewards) // 10)
                moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
                x_ma = np.arange(window_size-1, len(episode_rewards))
                plt.plot(x_ma, moving_avg, 'r-', linewidth=2, label=f'Moving Avg ({window_size} episodes)')

            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title(f'{env_name} - Episode Rewards')
            plt.grid(True, alpha=0.3)
            plt.legend()

            rewards_file = os.path.join(save_dir, f'{env_name}_episode_rewards.png')
            plt.savefig(rewards_file, dpi=300, bbox_inches='tight')
            all_saved_files.append(rewards_file)
            if not show_plots:
                plt.close()

        # 2. Training loss plot
        if losses and len(losses) > 0:
            plt.figure(figsize=(12, 6))
            plt.plot(losses, alpha=0.7, linewidth=1)
            plt.xlabel('Training Step')
            plt.ylabel('Loss')
            plt.title(f'{env_name} - Training Loss')
            plt.grid(True, alpha=0.3)

            loss_file = os.path.join(save_dir, f'{env_name}_training_loss.png')
            plt.savefig(loss_file, dpi=300, bbox_inches='tight')
            all_saved_files.append(loss_file)
            if not show_plots:
                plt.close()

        # 3. Episode lengths plot
        if episode_lengths and len(episode_lengths) > 0:
            plt.figure(figsize=(12, 6))
            plt.plot(episode_lengths, alpha=0.7, linewidth=1)
            plt.xlabel('Episode')
            plt.ylabel('Episode Length')
            plt.title(f'{env_name} - Episode Lengths')
            plt.grid(True, alpha=0.3)

            lengths_file = os.path.join(save_dir, f'{env_name}_episode_lengths.png')
            plt.savefig(lengths_file, dpi=300, bbox_inches='tight')
            all_saved_files.append(lengths_file)
            if not show_plots:
                plt.close()

        # 4. Epsilon decay plot
        if epsilon_values and len(epsilon_values) > 0:
            plt.figure(figsize=(12, 6))
            plt.plot(epsilon_values, alpha=0.7, linewidth=1)
            plt.xlabel('Episode')
            plt.ylabel('Epsilon')
            plt.title(f'{env_name} - Epsilon Decay')
            plt.grid(True, alpha=0.3)

            epsilon_file = os.path.join(save_dir, f'{env_name}_epsilon_decay.png')
            plt.savefig(epsilon_file, dpi=300, bbox_inches='tight')
            all_saved_files.append(epsilon_file)
            if not show_plots:
                plt.close()

        # 5. Evaluation rewards plot
        if eval_episodes and eval_rewards and len(eval_episodes) > 0:
            plt.figure(figsize=(12, 6))
            plt.plot(eval_episodes, eval_rewards, 'go-', linewidth=2, markersize=8)
            plt.xlabel('Episode')
            plt.ylabel('Evaluation Reward')
            plt.title(f'{env_name} - Evaluation Performance')
            plt.grid(True, alpha=0.3)

            eval_file = os.path.join(save_dir, f'{env_name}_evaluation_rewards.png')
            plt.savefig(eval_file, dpi=300, bbox_inches='tight')
            all_saved_files.append(eval_file)
            if not show_plots:
                plt.close()

        # 6. Reward distribution histogram
        if episode_rewards and len(episode_rewards) > 0:
            plt.figure(figsize=(12, 6))
            plt.hist(episode_rewards, bins=50, alpha=0.7, edgecolor='black')
            plt.xlabel('Reward')
            plt.ylabel('Frequency')
            plt.title(f'{env_name} - Reward Distribution')
            plt.grid(True, alpha=0.3)
            plt.axvline(np.mean(episode_rewards), color='r', linestyle='--',
                       label=f'Mean: {np.mean(episode_rewards):.2f}')
            plt.legend()

            hist_file = os.path.join(save_dir, f'{env_name}_reward_histogram.png')
            plt.savefig(hist_file, dpi=300, bbox_inches='tight')
            all_saved_files.append(hist_file)
            if not show_plots:
                plt.close()

        # 6b. Reward boxplot
        if episode_rewards and len(episode_rewards) > 10:
            plt.figure(figsize=(8, 6))
            plt.boxplot(episode_rewards, vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7),
                       medianprops=dict(color='red', linewidth=2))
            plt.ylabel('Reward')
            plt.title(f'{env_name} - Reward Boxplot')
            plt.grid(True, alpha=0.3)

            # Add statistics text
            stats_text = f'Mean: {np.mean(episode_rewards):.2f}\nStd: {np.std(episode_rewards):.2f}'
            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

            boxplot_file = os.path.join(save_dir, f'{env_name}_reward_boxplot.png')
            plt.savefig(boxplot_file, dpi=300, bbox_inches='tight')
            all_saved_files.append(boxplot_file)
            if not show_plots:
                plt.close()

        # 7. Performance summary plot
        if episode_rewards and len(episode_rewards) > 10:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'{env_name} - Training Summary', fontsize=16)

            # Episode rewards with moving average
            ax1.plot(episode_rewards, alpha=0.3, color='blue', linewidth=0.5)
            if len(episode_rewards) > 100:
                window_size = min(100, len(episode_rewards) // 10)
                moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
                x_ma = np.arange(window_size-1, len(episode_rewards))
                ax1.plot(x_ma, moving_avg, 'b-', linewidth=2)
            ax1.set_title('Episode Rewards')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward')
            ax1.grid(True, alpha=0.3)

            # Reward histogram
            ax2.hist(episode_rewards, bins=30, alpha=0.7, edgecolor='black')
            ax2.axvline(np.mean(episode_rewards), color='r', linestyle='--')
            ax2.set_title('Reward Distribution')
            ax2.set_xlabel('Reward')
            ax2.set_ylabel('Frequency')
            ax2.grid(True, alpha=0.3)

            # Cumulative rewards
            cumulative_rewards = np.cumsum(episode_rewards)
            ax3.plot(cumulative_rewards, color='green', linewidth=2)
            ax3.set_title('Cumulative Rewards')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Cumulative Reward')
            ax3.grid(True, alpha=0.3)

            # Performance metrics
            recent_rewards = episode_rewards[-min(100, len(episode_rewards)):]
            metrics_text = f"""
            Final Reward: {episode_rewards[-1]:.2f}
            Mean Reward: {np.mean(episode_rewards):.2f}
            Std Reward: {np.std(episode_rewards):.2f}
            Max Reward: {np.max(episode_rewards):.2f}
            Min Reward: {np.min(episode_rewards):.2f}
            Recent Mean: {np.mean(recent_rewards):.2f}
            """
            ax4.text(0.1, 0.5, metrics_text, fontsize=12,
                    transform=ax4.transAxes, verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            ax4.set_title('Performance Metrics')
            ax4.axis('off')

            plt.tight_layout()
            summary_file = os.path.join(save_dir, f'{env_name}_training_summary.png')
            plt.savefig(summary_file, dpi=300, bbox_inches='tight')
            all_saved_files.append(summary_file)
            if not show_plots:
                plt.close()

        logger.info(f"Generated {len(all_saved_files)} plots for {env_name}")

    except Exception as e:
        logger.error(f"Error generating plots: {e}")
        raise

    return all_saved_files