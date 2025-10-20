"""
工具函数
"""
"""
工具函数
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


def save_model(path: str, model_dict: Dict[str, Any]):
    """
    保存模型

    Args:
        path: 保存路径
        model_dict: 模型字典
    """
    if not isinstance(path, str):
        raise TypeError(f"path必须是字符串类型，当前类型为: {type(path)}")

    if not path.strip():
        raise ValueError("path不能为空字符串")

    if not isinstance(model_dict, dict):
        raise TypeError(f"model_dict必须是字典类型，当前类型为: {type(model_dict)}")

    if not model_dict:
        logger.warning("model_dict为空字典")

    try:
        # 获取目录路径
        dir_path = os.path.dirname(path)

        # 如果目录路径不为空，则创建目录
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        # 保存到临时文件，避免保存失败导致原文件损坏
        temp_path = path + '.tmp'
        torch.save(model_dict, temp_path)

        # 原子性替换
        if os.path.exists(path):
            backup_path = path + '.backup'
            os.replace(path, backup_path)
            try:
                os.replace(temp_path, path)
                os.remove(backup_path)
            except Exception:
                # 如果替换失败，恢复备份
                os.replace(backup_path, path)
                raise
        else:
            os.replace(temp_path, path)

        logger.info(f"模型已保存到: {path}")

    except Exception as e:
        logger.error(f"保存模型时发生错误: {str(e)}")
        # 清理临时文件
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        raise


def load_model(path: str) -> Dict[str, Any]:
    """
    加载模型

    Args:
        path: 模型文件路径

    Returns:
        模型字典
    """
    if not isinstance(path, str):
        raise TypeError(f"path必须是字符串类型，当前类型为: {type(path)}")

    if not path.strip():
        raise ValueError("path不能为空字符串")

    if not os.path.exists(path):
        raise FileNotFoundError(f"模型文件不存在: {path}")

    if not os.path.isfile(path):
        raise ValueError(f"路径不是文件: {path}")

    try:
        model_dict = torch.load(path, map_location='cpu', weights_only=False)

        if not isinstance(model_dict, dict):
            logger.warning(f"加载的模型不是字典类型，而是: {type(model_dict)}")

        logger.info(f"成功加载模型: {path}")
        return model_dict

    except Exception as e:
        logger.error(f"加载模型时发生错误: {str(e)}")
        raise


def to_tensor(x, dtype=torch.float32, device='cpu'):
    """
    转换为tensor

    Args:
        x: 输入数据
        dtype: 数据类型
        device: 设备

    Returns:
        转换后的tensor
    """
    if x is None:
        raise ValueError("输入数据不能为None")

    # 验证dtype
    valid_dtypes = [torch.float16, torch.float32, torch.float64, 
                    torch.int8, torch.int16, torch.int32, torch.int64,
                    torch.uint8, torch.bool]
    if dtype not in valid_dtypes:
        raise ValueError(f"不支持的dtype: {dtype}")

    # 验证device
    if isinstance(device, str):
        if device not in ['cpu', 'cuda'] and not device.startswith('cuda:'):
            raise ValueError(f"不支持的device: {device}")
        if device.startswith('cuda') and not torch.cuda.is_available():
            logger.warning(f"CUDA不可用，将使用CPU代替")
            device = 'cpu'
    elif isinstance(device, torch.device):
        if device.type == 'cuda' and not torch.cuda.is_available():
            logger.warning(f"CUDA不可用，将使用CPU代替")
            device = torch.device('cpu')
    else:
        raise TypeError(f"device类型错误: {type(device)}")

    try:
        if isinstance(x, torch.Tensor):
            return x.to(dtype=dtype, device=device)

        # 处理numpy数组
        if isinstance(x, np.ndarray):
            tensor = torch.from_numpy(x)
            return tensor.to(dtype=dtype, device=device)

        # 处理列表、元组等
        return torch.tensor(x, dtype=dtype, device=device)

    except Exception as e:
        logger.error(f"转换为tensor时发生错误: {str(e)}")
        raise ValueError(f"无法将输入转换为tensor: {str(e)}")


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


def setup_plotting_style():
    """Setup beautiful plotting style"""
    # Set style
    style.use('seaborn-v0_8')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']

    # Color palette
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592E83', '#1B998B']
    return colors


def plot_training_curves(episode_rewards: List[float],
                        episode_lengths: List[int] = None,
                        losses: List[float] = None,
                        eval_episodes: List[int] = None,
                        eval_rewards: List[float] = None,
                        epsilon_values: List[float] = None,
                        save_dir: str = "plots",
                        env_name: str = "Environment",
                        show: bool = False):
    """
    Plot comprehensive training curves as separate figures

    Args:
        episode_rewards: List of episode rewards
        episode_lengths: List of episode lengths
        losses: List of training losses
        eval_episodes: List of episode numbers where evaluation was performed
        eval_rewards: List of evaluation rewards
        epsilon_values: List of epsilon values (exploration rate)
        save_dir: Directory to save plots
        env_name: Name of the environment for plot titles
        show: Whether to display plots

    Returns:
        List of file paths to saved plots
    """
    colors = setup_plotting_style()
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_files = []

    # Calculate moving averages
    window = min(100, len(episode_rewards) // 10 + 1)
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        ma_episodes = np.arange(window-1, len(episode_rewards))
    else:
        moving_avg = episode_rewards
        ma_episodes = np.arange(len(episode_rewards))

    # Plot 1: Episode Rewards
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    ax1.plot(episode_rewards, alpha=0.3, color=colors[0], linewidth=0.5, label='Raw Rewards')
    ax1.plot(ma_episodes, moving_avg, color=colors[0], linewidth=2.5, label=f'Moving Avg ({window})')
    if eval_episodes and eval_rewards:
        ax1.scatter(eval_episodes, eval_rewards, color=colors[2], s=80,
                   alpha=0.8, label='Evaluation', zorder=5, edgecolors='black', linewidth=1)

    ax1.set_xlabel('Episode', fontsize=14)
    ax1.set_ylabel('Reward', fontsize=14)
    ax1.set_title(f'Episode Rewards - {env_name}', fontsize=16, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Add statistics text - positioned to avoid legend overlap
    if len(episode_rewards) > 0:
        stats_text = f'Mean: {np.mean(episode_rewards[-100:]):.2f}\n'
        stats_text += f'Std: {np.std(episode_rewards[-100:]):.2f}\n'
        stats_text += f'Best: {max(episode_rewards):.2f}'
        ax1.text(0.98, 0.98, stats_text, transform=ax1.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=11)

    plt.tight_layout()
    filename = f"{env_name}_episode_rewards_{timestamp}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    logger.info(f"Episode rewards plot saved to: {filepath}")
    saved_files.append(filepath)
    if show:
        plt.show()
    else:
        plt.close()

    # Plot 2: Episode Lengths
    if episode_lengths:
        fig2, ax2 = plt.subplots(figsize=(12, 8))
        window_len = min(50, len(episode_lengths) // 10 + 1)
        if len(episode_lengths) >= window_len:
            ma_lengths = np.convolve(episode_lengths, np.ones(window_len)/window_len, mode='valid')
            ma_len_episodes = np.arange(window_len-1, len(episode_lengths))
            ax2.plot(episode_lengths, alpha=0.3, color=colors[1], linewidth=0.5, label='Raw Length')
            ax2.plot(ma_len_episodes, ma_lengths, color=colors[1], linewidth=2.5,
                     label=f'Moving Avg ({window_len})')
        else:
            ax2.plot(episode_lengths, color=colors[1], linewidth=2.5, label='Episode Length')

        ax2.set_xlabel('Episode', fontsize=14)
        ax2.set_ylabel('Episode Length', fontsize=14)
        ax2.set_title(f'Episode Lengths - {env_name}', fontsize=16, fontweight='bold')
        ax2.legend(loc='upper left', fontsize=12)
        ax2.grid(True, alpha=0.3)

        # Add statistics text - positioned to avoid legend overlap
        if len(episode_lengths) > 0:
            stats_text = f'Mean: {np.mean(episode_lengths[-100:]):.1f}\n'
            stats_text += f'Std: {np.std(episode_lengths[-100:]):.1f}'
            ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                    fontsize=11)

        plt.tight_layout()
        filename = f"{env_name}_episode_lengths_{timestamp}.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Episode lengths plot saved to: {filepath}")
        saved_files.append(filepath)
        if show:
            plt.show()
        else:
            plt.close()

    # Plot 3: Training Loss
    if losses:
        fig3, ax3 = plt.subplots(figsize=(12, 8))
        loss_window = min(100, len(losses) // 10 + 1)
        if len(losses) >= loss_window:
            ma_losses = np.convolve(losses, np.ones(loss_window)/loss_window, mode='valid')
            ma_loss_episodes = np.arange(loss_window-1, len(losses))
            ax3.plot(losses, alpha=0.3, color=colors[3], linewidth=0.5, label='Raw Loss')
            ax3.plot(ma_loss_episodes, ma_losses, color=colors[3], linewidth=2.5,
                     label=f'Moving Avg ({loss_window})')
        else:
            ax3.plot(losses, color=colors[3], linewidth=2.5, label='Training Loss')

        ax3.set_xlabel('Training Step', fontsize=14)
        ax3.set_ylabel('Loss', fontsize=14)
        ax3.set_title(f'Training Loss - {env_name}', fontsize=16, fontweight='bold')
        ax3.legend(loc='upper left', fontsize=12)
        ax3.grid(True, alpha=0.3)

        # Use log scale if losses vary widely
        if len(losses) > 0 and max(losses) / min(losses) > 100:
            ax3.set_yscale('log')
            ax3.set_title(f'Training Loss (Log Scale) - {env_name}', fontsize=16, fontweight='bold')

        # Add statistics text - positioned to avoid legend overlap
        if len(losses) > 0:
            stats_text = f'Final: {losses[-1]:.4f}\n'
            stats_text += f'Best: {min(losses):.4f}\n'
            stats_text += f'Steps: {len(losses)}'
            ax3.text(0.98, 0.98, stats_text, transform=ax3.transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                    fontsize=11)

        plt.tight_layout()
        filename = f"{env_name}_training_loss_{timestamp}.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Training loss plot saved to: {filepath}")
        saved_files.append(filepath)
        if show:
            plt.show()
        else:
            plt.close()

    # Plot 4: Epsilon (Exploration Rate)
    if epsilon_values:
        fig4, ax4 = plt.subplots(figsize=(12, 8))
        ax4.plot(epsilon_values, color=colors[4], linewidth=2.5)
        ax4.set_xlabel('Episode', fontsize=14)
        ax4.set_ylabel('Epsilon', fontsize=14)
        ax4.set_title(f'Exploration Rate (Epsilon) - {env_name}', fontsize=16, fontweight='bold')
        ax4.grid(True, alpha=0.3)

        # Add statistics text
        if len(epsilon_values) > 0:
            stats_text = f'Start: {epsilon_values[0]:.3f}\n'
            stats_text += f'End: {epsilon_values[-1]:.3f}\n'
            stats_text += f'Decay: {epsilon_values[0] * 0.995**len(epsilon_values):.3f}'
            ax4.text(0.02, 0.98, stats_text, transform=ax4.transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    fontsize=11)

        plt.tight_layout()
        filename = f"{env_name}_epsilon_decay_{timestamp}.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Epsilon decay plot saved to: {filepath}")
        saved_files.append(filepath)
        if show:
            plt.show()
        else:
            plt.close()

    return saved_files


def plot_reward_distribution(episode_rewards: List[float],
                           save_dir: str = "plots",
                           env_name: str = "Environment",
                           show: bool = False):
    """
    Plot reward distribution as separate figures

    Args:
        episode_rewards: List of episode rewards
        save_dir: Directory to save plots
        env_name: Name of the environment
        show: Whether to display plots

    Returns:
        List of file paths to saved plots
    """
    colors = setup_plotting_style()
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_files = []

    rewards_array = np.array(episode_rewards)

    # Plot 1: Reward Histogram
    fig1, ax1 = plt.subplots(figsize=(12, 8))

    # Calculate optimal number of bins
    if len(rewards_array) > 0:
        bins = min(50, max(20, int(np.sqrt(len(rewards_array)))))
    else:
        bins = 50

    n, bins_edges, patches = ax1.hist(rewards_array, bins=bins, alpha=0.7,
                                     color=colors[0], edgecolor='black', linewidth=0.5)

    # Add mean and median lines
    mean_val = np.mean(rewards_array)
    median_val = np.median(rewards_array)
    ax1.axvline(mean_val, color=colors[2], linestyle='--', linewidth=2.5,
               label=f'Mean: {mean_val:.2f}')
    ax1.axvline(median_val, color=colors[3], linestyle='--', linewidth=2.5,
               label=f'Median: {median_val:.2f}')

    ax1.set_xlabel('Reward', fontsize=14)
    ax1.set_ylabel('Frequency', fontsize=14)
    ax1.set_title(f'Reward Distribution Histogram - {env_name}', fontsize=16, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Add detailed statistics text - positioned to avoid legend overlap
    stats_text = f'Distribution Statistics:\n'
    stats_text += f'Mean: {mean_val:.2f}\n'
    stats_text += f'Median: {median_val:.2f}\n'
    stats_text += f'Std: {np.std(rewards_array):.2f}\n'
    stats_text += f'Min: {np.min(rewards_array):.2f}\n'
    stats_text += f'Max: {np.max(rewards_array):.2f}\n'
    stats_text += f'Samples: {len(rewards_array)}\n'
    stats_text += f'Skewness: {calculate_skewness(rewards_array):.3f}'

    ax1.text(0.02, 0.02, stats_text, transform=ax1.transAxes,
            verticalalignment='bottom', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=11)

    plt.tight_layout()
    filename = f"{env_name}_reward_histogram_{timestamp}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    logger.info(f"Reward histogram saved to: {filepath}")
    saved_files.append(filepath)
    if show:
        plt.show()
    else:
        plt.close()

    # Plot 2: Box Plot and Statistics
    fig2, ax2 = plt.subplots(figsize=(12, 8))

    # Create box plot
    box_plot = ax2.boxplot(rewards_array, patch_artist=True,
                          boxprops=dict(facecolor=colors[1], alpha=0.7),
                          medianprops=dict(color=colors[2], linewidth=2.5),
                          whiskerprops=dict(linewidth=2),
                          capprops=dict(linewidth=2))

    ax2.set_ylabel('Reward', fontsize=14)
    ax2.set_title(f'Reward Box Plot - {env_name}', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Add comprehensive statistics text
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    percentile_values = [np.percentile(rewards_array, p) for p in percentiles]

    stats_text = f'Reward Percentiles:\n'
    for p, v in zip(percentiles, percentile_values):
        stats_text += f'{p:2d}th: {v:7.2f}\n'
    stats_text += f'\nAdditional Stats:\n'
    stats_text += f'Mean:     {mean_val:7.2f}\n'
    stats_text += f'Std:      {np.std(rewards_array):7.2f}\n'
    stats_text += f'IQR:      {percentile_values[4] - percentile_values[2]:7.2f}'

    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
            fontsize=11, family='monospace')

    plt.tight_layout()
    filename = f"{env_name}_reward_boxplot_{timestamp}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    logger.info(f"Reward boxplot saved to: {filepath}")
    saved_files.append(filepath)
    if show:
        plt.show()
    else:
        plt.close()

    return saved_files


def calculate_skewness(data):
    """Calculate skewness of data"""
    if len(data) < 2:
        return 0.0
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0.0
    return np.mean(((data - mean) / std) ** 3)


def plot_training_summary(episode_rewards: List[float],
                         episode_lengths: List[int] = None,
                         losses: List[float] = None,
                         eval_rewards: List[float] = None,
                         save_dir: str = "plots",
                         env_name: str = "Environment",
                         config: Dict[str, Any] = None,
                         show: bool = False):
    """
    Create comprehensive training summary as separate figures

    Args:
        episode_rewards: List of episode rewards
        episode_lengths: List of episode lengths
        losses: List of training losses
        eval_rewards: List of evaluation rewards
        save_dir: Directory to save plots
        env_name: Name of the environment
        config: Training configuration dictionary
        show: Whether to display plots

    Returns:
        List of file paths to saved plots
    """
    colors = setup_plotting_style()
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_files = []

    # Plot 1: Training Summary Statistics
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    ax1.axis('off')

    info_text = f"Training Summary - {env_name}\n" + "="*40 + "\n\n"
    info_text += f"PERFORMANCE METRICS:\n"
    info_text += f"  Total Episodes: {len(episode_rewards):,}\n"
    info_text += f"  Final Reward: {episode_rewards[-1]:.2f}\n"
    info_text += f"  Best Reward: {max(episode_rewards):.2f}\n"
    info_text += f"  Worst Reward: {min(episode_rewards):.2f}\n"
    info_text += f"  Mean Reward (last 100): {np.mean(episode_rewards[-100:]):.2f}\n"
    info_text += f"  Std Dev (last 100): {np.std(episode_rewards[-100:]):.2f}\n"
    info_text += f"  Median Reward: {np.median(episode_rewards):.2f}\n"

    if episode_lengths:
        info_text += f"  Avg Episode Length: {np.mean(episode_lengths[-100:]):.1f}\n"

    if eval_rewards:
        info_text += f"  Best Eval Reward: {max(eval_rewards):.2f}\n"
        info_text += f"  Latest Eval Reward: {eval_rewards[-1]:.2f}\n"

    info_text += f"\nTRAINING CONFIGURATION:\n"
    if config:
        info_text += f"  Learning Rate: {config.get('algorithm', {}).get('lr', 'N/A')}\n"
        info_text += f"  Batch Size: {config.get('training', {}).get('batch_size', 'N/A')}\n"
        info_text += f"  Buffer Size: {config.get('training', {}).get('buffer_size', 'N/A')}\n"
        info_text += f"  Total Episodes: {config.get('training', {}).get('total_episodes', 'N/A')}\n"
        info_text += f"  Gamma: {config.get('algorithm', {}).get('gamma', 'N/A')}\n"

    # Learning efficiency
    if len(episode_rewards) > 100:
        early_mean = np.mean(episode_rewards[:min(50, len(episode_rewards))])
        late_mean = np.mean(episode_rewards[-min(50, len(episode_rewards)):])
        improvement = late_mean - early_mean
        info_text += f"\nLEARNING EFFICIENCY:\n"
        info_text += f"  Early Mean (first 50): {early_mean:.2f}\n"
        info_text += f"  Late Mean (last 50): {late_mean:.2f}\n"
        info_text += f"  Improvement: {improvement:+.2f} ({improvement/abs(early_mean)*100:+.1f}%)\n"

    ax1.text(0.05, 0.95, info_text, transform=ax1.transAxes,
            verticalalignment='top', fontsize=12, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()
    filename = f"{env_name}_training_summary_{timestamp}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    logger.info(f"Training summary saved to: {filepath}")
    saved_files.append(filepath)
    if show:
        plt.show()
    else:
        plt.close()

    # Plot 2: Performance Over Time (chunks)
    if len(episode_rewards) > 100:
        fig2, ax2 = plt.subplots(figsize=(12, 8))

        # Calculate performance in chunks
        chunk_size = max(10, len(episode_rewards) // 20)
        chunks = [episode_rewards[i:i+chunk_size] for i in range(0, len(episode_rewards), chunk_size)]
        chunk_means = [np.mean(chunk) for chunk in chunks]
        chunk_stds = [np.std(chunk) if len(chunk) > 1 else 0 for chunk in chunks]
        chunk_episodes = [i * chunk_size + chunk_size//2 for i in range(len(chunks))]

        # Plot with error bars
        ax2.errorbar(chunk_episodes, chunk_means, yerr=chunk_stds,
                    fmt='o-', color=colors[3], linewidth=2.5, markersize=6,
                    capsize=4, capthick=2, label='Chunk Mean ± Std')

        ax2.set_xlabel('Episode', fontsize=14)
        ax2.set_ylabel('Average Reward', fontsize=14)
        ax2.set_title(f'Performance Trend Over Time - {env_name}', fontsize=16, fontweight='bold')
        ax2.legend(loc='best', fontsize=12)
        ax2.grid(True, alpha=0.3)

        # Add trend line
        if len(chunk_episodes) > 1:
            z = np.polyfit(chunk_episodes, chunk_means, 1)
            p = np.poly1d(z)
            ax2.plot(chunk_episodes, p(chunk_episodes), '--', color=colors[4], linewidth=2,
                    label=f'Trend: {z[0]:.4f}x + {z[1]:.2f}')
            ax2.legend(loc='best', fontsize=12)

        plt.tight_layout()
        filename = f"{env_name}_performance_trend_{timestamp}.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Performance trend saved to: {filepath}")
        saved_files.append(filepath)
        if show:
            plt.show()
        else:
            plt.close()

    # Plot 3: Learning Curves (Different Smoothing Windows)
    if len(episode_rewards) > 50:
        fig3, ax3 = plt.subplots(figsize=(12, 8))

        window_sizes = [10, 25, 50, 100]
        for i, window in enumerate(window_sizes):
            if len(episode_rewards) >= window:
                moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
                ma_episodes = np.arange(window-1, len(episode_rewards))
                alpha = 0.4 + 0.4 * (i / len(window_sizes))
                ax3.plot(ma_episodes, moving_avg, alpha=alpha,
                        linewidth=2 + i*0.5, color=colors[i % len(colors)],
                        label=f'Moving Avg (window={window})')

        ax3.set_xlabel('Episode', fontsize=14)
        ax3.set_ylabel('Reward', fontsize=14)
        ax3.set_title(f'Learning Curves - {env_name}', fontsize=16, fontweight='bold')
        ax3.legend(loc='best', fontsize=12)
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        filename = f"{env_name}_learning_curves_{timestamp}.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Learning curves saved to: {filepath}")
        saved_files.append(filepath)
        if show:
            plt.show()
        else:
            plt.close()

    # Plot 4: Recent Performance Distribution
    if len(episode_rewards) > 50:
        fig4, ax4 = plt.subplots(figsize=(12, 8))

        recent_rewards = episode_rewards[-500:] if len(episode_rewards) > 500 else episode_rewards

        # Create histogram
        n, bins_edges, patches = ax4.hist(recent_rewards, bins=30, alpha=0.7,
                                         color=colors[5], edgecolor='black', linewidth=0.5)

        # Add statistics lines
        recent_mean = np.mean(recent_rewards)
        recent_median = np.median(recent_rewards)
        ax4.axvline(recent_mean, color=colors[2], linestyle='--', linewidth=2.5,
                   label=f'Mean: {recent_mean:.2f}')
        ax4.axvline(recent_median, color=colors[3], linestyle='--', linewidth=2.5,
                   label=f'Median: {recent_median:.2f}')

        ax4.set_xlabel('Reward', fontsize=14)
        ax4.set_ylabel('Frequency', fontsize=14)
        ax4.set_title(f'Recent Performance Distribution (last {len(recent_rewards)} episodes) - {env_name}',
                     fontsize=16, fontweight='bold')
        ax4.legend(loc='best', fontsize=12)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        filename = f"{env_name}_recent_distribution_{timestamp}.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Recent distribution saved to: {filepath}")
        saved_files.append(filepath)
        if show:
            plt.show()
        else:
            plt.close()

    return saved_files


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
    Generate and save all training plots as separate figures

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
        # 1. Training curves (now returns multiple files)
        training_curve_files = plot_training_curves(
            episode_rewards, episode_lengths, losses,
            eval_episodes, eval_rewards, epsilon_values,
            save_dir, env_name, show_plots
        )
        all_saved_files.extend(training_curve_files)

        # 2. Reward distribution (now returns multiple files)
        distribution_files = plot_reward_distribution(
            episode_rewards, save_dir, env_name, show_plots
        )
        all_saved_files.extend(distribution_files)

        # 3. Training summary (now returns multiple files)
        summary_files = plot_training_summary(
            episode_rewards, episode_lengths, losses, eval_rewards,
            save_dir, env_name, config, show_plots
        )
        all_saved_files.extend(summary_files)

        logger.info(f"All {len(all_saved_files)} plots saved to directory: {save_dir}")

        # Create an index file listing all plots
        create_plot_index(all_saved_files, save_dir, env_name)

    except Exception as e:
        logger.error(f"Error generating plots: {e}")
        raise

    return all_saved_files


def create_plot_index(plot_files: List[str], save_dir: str, env_name: str):
    """
    Create an index file listing all generated plots with descriptions

    Args:
        plot_files: List of file paths to saved plots
        save_dir: Directory where plots are saved
        env_name: Environment name
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    index_file = os.path.join(save_dir, f"{env_name}_plot_index_{timestamp}.txt")

    try:
        with open(index_file, 'w', encoding='utf-8') as f:
            f.write(f"Training Plots Index - {env_name}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total plots: {len(plot_files)}\n\n")

            # Categorize plots by type
            categories = {
                'Training Progress': [],
                'Statistical Analysis': [],
                'Performance Analysis': [],
                'Configuration': []
            }

            for plot_file in plot_files:
                filename = os.path.basename(plot_file)
                if 'episode_rewards' in filename or 'episode_lengths' in filename or 'training_loss' in filename or 'epsilon_decay' in filename:
                    categories['Training Progress'].append(filename)
                elif 'reward_histogram' in filename or 'reward_boxplot' in filename or 'distribution' in filename:
                    categories['Statistical Analysis'].append(filename)
                elif 'learning_curves' in filename or 'performance_trend' in filename:
                    categories['Performance Analysis'].append(filename)
                elif 'summary' in filename:
                    categories['Configuration'].append(filename)

            for category, files in categories.items():
                if files:
                    f.write(f"\n{category}:\n")
                    f.write("-" * len(category) + "\n")
                    for i, file in enumerate(files, 1):
                        f.write(f"{i}. {file}\n")

            f.write(f"\n" + "=" * 50 + "\n")
            f.write("Note: All plots are saved in high resolution (300 DPI)\n")

        logger.info(f"Plot index created: {index_file}")

    except Exception as e:
        logger.warning(f"Could not create plot index file: {e}")


def load_training_data(filepath: str) -> Dict[str, Any]:
    """
    加载训练数据
    
    Args:
        filepath: 训练数据文件路径
        
    Returns:
        训练数据字典
    """
    import json
    
    if not isinstance(filepath, str):
        raise TypeError(f"filepath必须是字符串类型，当前类型为: {type(filepath)}")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"数据文件不存在: {filepath}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"成功加载训练数据: {filepath}")
        return data
    
    except json.JSONDecodeError as e:
        logger.error(f"JSON解析错误: {str(e)}")
        raise ValueError(f"数据文件格式错误: {str(e)}")
    except Exception as e:
        logger.error(f"加载数据文件时发生错误: {str(e)}")
        raise


def plot_from_file(filepath: str, 
                   save_dir: str = "plots",
                   show_plots: bool = False) -> List[str]:
    """
    从保存的文件加载数据并绘制所有图表
    
    Args:
        filepath: 训练数据文件路径
        save_dir: 图表保存目录
        show_plots: 是否显示图表
        
    Returns:
        保存的图表文件路径列表
        
    Example:
        >>> plot_files = plot_from_file('checkpoints/DEM_normal_training_data_20231015_123456.json')
        >>> print(f"生成了 {len(plot_files)} 个图表")
    """
    # 加载数据
    data = load_training_data(filepath)
    
    # 提取指标
    metrics = data['metrics']
    config = data.get('config', {})
    env_name = data.get('environment', {}).get('name', 'Environment')
    difficulty = data.get('environment', {}).get('difficulty', '')
    
    # 生成环境名称
    if difficulty:
        full_env_name = f"{env_name}_{difficulty}"
    else:
        full_env_name = env_name
    
    print(f"\n📊 开始从数据文件绘制图表...")
    print(f"   环境: {full_env_name}")
    print(f"   总 episodes: {data.get('total_episodes', len(metrics['episode_rewards']))}")
    print(f"   保存目录: {save_dir}")
    
    # 绘制所有图表
    plot_files = save_training_plots(
        episode_rewards=metrics['episode_rewards'],
        episode_lengths=metrics.get('episode_lengths'),
        losses=metrics.get('losses'),
        eval_episodes=metrics.get('eval_episodes'),
        eval_rewards=metrics.get('eval_rewards'),
        epsilon_values=metrics.get('epsilon_history'),
        save_dir=save_dir,
        env_name=full_env_name,
        config=config,
        show_plots=show_plots
    )
    
    print(f"\n✅ 绘图完成! 生成了 {len(plot_files)} 个图表")
    print(f"📁 图表保存在: {save_dir}")
    
    return plot_files


def list_training_data_files(directory: str = "checkpoints") -> List[str]:
    """
    列出目录中所有的训练数据文件
    
    Args:
        directory: 查找目录
        
    Returns:
        训练数据文件路径列表
    """
    import glob
    
    if not os.path.exists(directory):
        logger.warning(f"目录不存在: {directory}")
        return []
    
    pattern = os.path.join(directory, "*_training_data_*.json")
    files = glob.glob(pattern)
    files.sort(key=os.path.getmtime, reverse=True)  # 按修改时间排序
    
    return files


def print_training_data_summary(filepath: str):
    """
    打印训练数据摘要
    
    Args:
        filepath: 训练数据文件路径
    """
    data = load_training_data(filepath)
    
    print(f"\n{'='*60}")
    print(f"训练数据摘要: {os.path.basename(filepath)}")
    print(f"{'='*60}")
    
    # 环境信息
    env_info = data.get('environment', {})
    print(f"\n🎮 环境信息:")
    print(f"   名称: {env_info.get('name', 'N/A')}")
    print(f"   难度: {env_info.get('difficulty', 'N/A')}")
    print(f"   智能体数: {env_info.get('n_agents', 'N/A')}")
    
    # 训练信息
    metrics = data.get('metrics', {})
    episode_rewards = metrics.get('episode_rewards', [])
    
    print(f"\n📊 训练统计:")
    print(f"   总 episodes: {len(episode_rewards)}")
    print(f"   时间戳: {data.get('timestamp', 'N/A')}")
    
    if episode_rewards:
        print(f"\n🎯 性能指标:")
        print(f"   平均奖励: {np.mean(episode_rewards):.2f}")
        print(f"   最佳奖励: {max(episode_rewards):.2f}")
        print(f"   最终奖励: {episode_rewards[-1]:.2f}")
        print(f"   标准差: {np.std(episode_rewards):.2f}")
        
        # 最后100个episodes的统计
        if len(episode_rewards) >= 100:
            recent_rewards = episode_rewards[-100:]
            print(f"\n   最近100 episodes:")
            print(f"      平均: {np.mean(recent_rewards):.2f}")
            print(f"      最佳: {max(recent_rewards):.2f}")
    
    # 评估结果
    eval_rewards = metrics.get('eval_rewards', [])
    if eval_rewards:
        print(f"\n🎯 评估结果:")
        print(f"   评估次数: {len(eval_rewards)}")
        print(f"   平均分数: {np.mean(eval_rewards):.2f}")
        print(f"   最佳分数: {max(eval_rewards):.2f}")
    
    print(f"\n{'='*60}\n")