"""
VDN 工具函数
"""
import torch
import numpy as np
import random
import yaml
import os
import sys
import json
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


def save_training_data(data: Dict[str, Any], filepath: str):
    """
    保存训练数据

    Args:
        data: 训练数据
        filepath: 保存路径
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # 添加时间戳
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().isoformat()

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"训练数据已保存到: {filepath}")

    except Exception as e:
        logger.error(f"保存训练数据时发生错误: {str(e)}")
        raise


def load_training_data(filepath: str) -> Dict[str, Any]:
    """
    加载训练数据

    Args:
        filepath: 训练数据文件路径

    Returns:
        训练数据字典
    """
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
    try:
        style.use('seaborn-v0_8')
    except:
        style.use('default')

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
            f.write(f"VDN Training Plots Index - {env_name}\n")
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

        logger.info(f"VDN plot index created: {index_file}")

    except Exception as e:
        logger.warning(f"Could not create plot index file: {e}")


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
    生成并保存VDN训练图表（与QMIX格式完全一致）

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
    try:
        # 尝试从QMIX导入绘图工具
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../qmix/src'))
        from utils import plot_training_curves, plot_reward_distribution, plot_training_summary

        all_saved_files = []

        # 1. Training curves (返回多个文件)
        training_curve_files = plot_training_curves(
            episode_rewards, episode_lengths, losses,
            eval_episodes, eval_rewards, epsilon_values,
            save_dir, env_name, show_plots
        )
        all_saved_files.extend(training_curve_files)

        # 2. Reward distribution (返回多个文件)
        distribution_files = plot_reward_distribution(
            episode_rewards, save_dir, env_name, show_plots
        )
        all_saved_files.extend(distribution_files)

        # 3. Training summary (返回多个文件)
        summary_files = plot_training_summary(
            episode_rewards, episode_lengths, losses, eval_rewards,
            save_dir, env_name, config, show_plots
        )
        all_saved_files.extend(summary_files)

        logger.info(f"VDN训练图表已保存到目录: {save_dir}")
        logger.info(f"成功生成 {len(all_saved_files)} 个图表")

        # 创建图表索引文件
        create_plot_index(all_saved_files, save_dir, env_name)

        return all_saved_files

    except ImportError:
        logger.warning("无法导入QMIX绘图工具，使用简化版绘图")
        return save_simple_plots(episode_rewards, episode_lengths, losses, save_dir, env_name, show_plots)
    except Exception as e:
        logger.error(f"生成图表时发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return save_simple_plots(episode_rewards, episode_lengths, losses, save_dir, env_name, show_plots)


def save_simple_plots(episode_rewards: List[float],
                      episode_lengths: List[int] = None,
                      losses: List[float] = None,
                      save_dir: str = "plots",
                      env_name: str = "Environment",
                      show_plots: bool = False):
    """
    保存简化版图表

    Args:
        episode_rewards: List of episode rewards
        episode_lengths: List of episode lengths
        losses: List of training losses
        save_dir: Directory to save plots
        env_name: Name of the environment
        show_plots: Whether to display plots

    Returns:
        List of file paths to saved plots
    """
    colors = setup_plotting_style()
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_files = []

    try:
        # Plot episode rewards
        plt.figure(figsize=(12, 8))
        plt.plot(episode_rewards, alpha=0.7, color=colors[0], linewidth=1, label='Episode Rewards')

        # Add moving average
        if len(episode_rewards) >= 10:
            window = min(100, len(episode_rewards) // 10)
            moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
            ma_episodes = np.arange(window-1, len(episode_rewards))
            plt.plot(ma_episodes, moving_avg, color=colors[1], linewidth=2.5, label=f'Moving Avg ({window})')

        plt.xlabel('Episode', fontsize=14)
        plt.ylabel('Reward', fontsize=14)
        plt.title(f'VDN Episode Rewards - {env_name}', fontsize=16, fontweight='bold')
        plt.legend(loc='best', fontsize=12)
        plt.grid(True, alpha=0.3)

        # Add statistics
        if len(episode_rewards) > 0:
            stats_text = f'Mean: {np.mean(episode_rewards[-100:]):.2f}\n'
            stats_text += f'Best: {max(episode_rewards):.2f}'
            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    fontsize=11)

        plt.tight_layout()
        filename = f"VDN_{env_name}_rewards_{timestamp}.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        saved_files.append(filepath)

        if show_plots:
            plt.show()
        else:
            plt.close()

        # Plot training loss if available
        if losses:
            plt.figure(figsize=(12, 8))
            plt.plot(losses, color=colors[2], linewidth=1, alpha=0.7, label='Training Loss')

            if len(losses) >= 10:
                window = min(100, len(losses) // 10)
                ma_losses = np.convolve(losses, np.ones(window)/window, mode='valid')
                ma_episodes = np.arange(window-1, len(losses))
                plt.plot(ma_episodes, ma_losses, color=colors[3], linewidth=2.5, label=f'Moving Avg ({window})')

            plt.xlabel('Training Step', fontsize=14)
            plt.ylabel('Loss', fontsize=14)
            plt.title(f'VDN Training Loss - {env_name}', fontsize=16, fontweight='bold')
            plt.legend(loc='best', fontsize=12)
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            filename = f"VDN_{env_name}_loss_{timestamp}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            saved_files.append(filepath)

            if show_plots:
                plt.show()
            else:
                plt.close()

        logger.info(f"简化版VDN图表已保存到: {save_dir}")
        return saved_files

    except Exception as e:
        logger.error(f"保存简化版图表时发生错误: {str(e)}")
        return []


def create_experiment_name(env_name: str, difficulty: str, algorithm: str = "VDN") -> str:
    """
    创建实验名称

    Args:
        env_name: 环境名称
        difficulty: 难度级别
        algorithm: 算法名称

    Returns:
        实验名称
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{algorithm}_{env_name}_{difficulty}_{timestamp}"


def format_time(seconds: float) -> str:
    """
    格式化时间

    Args:
        seconds: 秒数

    Returns:
        格式化的时间字符串
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{int(minutes)}m {seconds:.0f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)}h {int(minutes)}m"


def calculate_vdn_statistics(episode_rewards: List[float],
                            window: int = 100) -> Dict[str, float]:
    """
    计算VDN训练统计信息

    Args:
        episode_rewards: Episode奖励列表
        window: 统计窗口大小

    Returns:
        统计信息字典
    """
    if not episode_rewards:
        return {
            'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
            'median': 0.0, 'final': 0.0, 'improvement': 0.0
        }

    rewards = np.array(episode_rewards)

    stats = {
        'mean': float(np.mean(rewards)),
        'std': float(np.std(rewards)),
        'min': float(np.min(rewards)),
        'max': float(np.max(rewards)),
        'median': float(np.median(rewards)),
        'final': float(rewards[-1])
    }

    # 计算改进率
    if len(rewards) >= window:
        early_mean = np.mean(rewards[:window])
        late_mean = np.mean(rewards[-window:])
        stats['improvement'] = float(late_mean - early_mean)
        stats['improvement_percent'] = float((late_mean - early_mean) / abs(early_mean) * 100) if early_mean != 0 else 0.0

    return stats


def validate_config(config: Dict[str, Any]) -> bool:
    """
    验证VDN配置文件

    Args:
        config: 配置字典

    Returns:
        是否有效
    """
    required_sections = ['env', 'algorithm', 'model', 'training', 'exploration']

    # 检查必需的配置节
    for section in required_sections:
        if section not in config:
            logger.error(f"配置文件缺少必需的节: {section}")
            return False

    # 检查环境配置
    env_config = config['env']
    if 'name' not in env_config:
        logger.error("环境配置缺少'name'字段")
        return False

    # 检查算法配置
    algo_config = config['algorithm']
    required_algo_fields = ['gamma', 'learning_rate']
    for field in required_algo_fields:
        if field not in algo_config:
            logger.error(f"算法配置缺少必需字段: {field}")
            return False

    # 检查模型配置
    model_config = config['model']
    if 'hidden_dim' not in model_config:
        logger.error("模型配置缺少'hidden_dim'字段")
        return False

    # 检查训练配置
    training_config = config['training']
    required_training_fields = ['total_episodes', 'batch_size']
    for field in required_training_fields:
        if field not in training_config:
            logger.error(f"训练配置缺少必需字段: {field}")
            return False

    logger.info("配置文件验证通过")
    return True


def log_experiment_info(config: Dict[str, Any], env_info: Dict[str, Any]):
    """
    记录实验信息

    Args:
        config: 配置字典
        env_info: 环境信息字典
    """
    logger.info("="*60)
    logger.info("VDN 实验配置")
    logger.info("="*60)

    # 环境信息
    env_config = config['env']
    logger.info(f"环境: {env_config['name']}")
    logger.info(f"难度: {env_config['difficulty']}")
    logger.info(f"智能体数量: {env_info['n_agents']}")
    logger.info(f"观测维度: {env_info['obs_dims']}")
    logger.info(f"动作维度: {env_info['act_dims']}")

    # 算法配置
    algo_config = config['algorithm']
    logger.info(f"折扣因子: {algo_config['gamma']}")
    logger.info(f"学习率: {algo_config['learning_rate']}")

    # 训练配置
    training_config = config['training']
    logger.info(f"总Episodes: {training_config['total_episodes']}")
    logger.info(f"批次大小: {training_config['batch_size']}")
    logger.info(f"缓冲区大小: {training_config['buffer_size']}")

    logger.info("="*60)