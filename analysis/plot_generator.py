"""
Plot Generator for MARL Training Analysis
Beautiful and publication-ready visualization
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Set publication-quality style
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 18,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'lines.linewidth': 2,
    'axes.prop_cycle': plt.cycler(color=sns.color_palette("husl", 8))
})


class PlotGenerator:
    """Generate beautiful plots for MARL training analysis"""
    
    def __init__(self, output_dir: str = "plots", style: str = "seaborn-v0_8-darkgrid"):
        """
        Initialize plot generator
        
        Args:
            output_dir: Directory to save plots
            style: Matplotlib style
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to set style
        try:
            plt.style.use(style)
        except:
            logger.warning(f"Style {style} not available, using default")
        
        # Color palette
        self.colors = {
            'QMIX': '#E74C3C',    # Red
            'IQL': '#3498DB',     # Blue
            'VDN': '#2ECC71',     # Green
            'COMA': '#F39C12',    # Orange
        }
        
        self.linestyles = {
            'QMIX': '-',
            'IQL': '--',
            'VDN': '-.',
            'COMA': ':'
        }
    
    def smooth_curve(self, values: np.ndarray, window: int = 50) -> np.ndarray:
        """Apply moving average smoothing"""
        if len(values) < window:
            return values
        return np.convolve(values, np.ones(window) / window, mode='valid')
    
    def plot_learning_curves(
        self,
        data: Dict[str, Dict[str, List[float]]],
        environment: str,
        smooth_window: int = 50,
        save: bool = True
    ) -> plt.Figure:
        """
        Plot learning curves for all algorithms on one environment
        
        Args:
            data: {algorithm: {env: rewards}}
            environment: Environment name
            smooth_window: Window size for smoothing
            save: Whether to save the figure
            
        Returns:
            matplotlib Figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        for algo in data:
            if environment not in data[algo]:
                continue
            
            rewards = np.array(data[algo][environment])
            episodes = np.arange(len(rewards))
            
            color = self.colors.get(algo, '#000000')
            linestyle = self.linestyles.get(algo, '-')
            
            # Plot 1: Raw + Smoothed
            ax1.plot(episodes, rewards, alpha=0.2, color=color)
            smoothed = self.smooth_curve(rewards, smooth_window)
            smooth_episodes = episodes[:len(smoothed)]
            ax1.plot(smooth_episodes, smoothed, label=algo, 
                    color=color, linestyle=linestyle, linewidth=2.5)
            
            # Plot 2: Only smoothed (cleaner)
            ax2.plot(smooth_episodes, smoothed, label=algo,
                    color=color, linestyle=linestyle, linewidth=2.5)
            
            # Add confidence interval (std of windows)
            if len(rewards) > smooth_window:
                std_values = []
                for i in range(len(smoothed)):
                    window_data = rewards[i:i+smooth_window]
                    std_values.append(np.std(window_data))
                std_values = np.array(std_values)
                
                ax2.fill_between(
                    smooth_episodes,
                    smoothed - std_values,
                    smoothed + std_values,
                    alpha=0.2,
                    color=color
                )
        
        # Styling
        env_title = environment.replace('_', ' ').title()
        
        ax1.set_xlabel('Episode', fontweight='bold')
        ax1.set_ylabel('Episode Reward', fontweight='bold')
        ax1.set_title(f'Learning Curves - {env_title}', fontweight='bold')
        ax1.legend(loc='best', framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Episode', fontweight='bold')
        ax2.set_ylabel('Smoothed Episode Reward', fontweight='bold')
        ax2.set_title(f'Smoothed Curves with Std - {env_title}', fontweight='bold')
        ax2.legend(loc='best', framealpha=0.9)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filename = self.output_dir / f"learning_curves_{environment}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot: {filename}")
        
        return fig
    
    def plot_final_performance_comparison(
        self,
        metrics: Dict[str, Dict[str, Dict]],
        save: bool = True
    ) -> plt.Figure:
        """
        Bar chart comparing final performance across environments
        
        Args:
            metrics: {algorithm: {environment: analysis_results}}
            save: Whether to save
            
        Returns:
            matplotlib Figure
        """
        # Organize data
        environments = set()
        for algo in metrics:
            environments.update(metrics[algo].keys())
        environments = sorted(list(environments))
        
        algorithms = sorted(metrics.keys())
        
        # Extract final performance
        data_matrix = np.zeros((len(algorithms), len(environments)))
        std_matrix = np.zeros((len(algorithms), len(environments)))
        
        for i, algo in enumerate(algorithms):
            for j, env in enumerate(environments):
                if env in metrics[algo]:
                    fp = metrics[algo][env]['final_performance']
                    data_matrix[i, j] = fp['mean']
                    std_matrix[i, j] = fp['std']
                else:
                    data_matrix[i, j] = np.nan
                    std_matrix[i, j] = np.nan
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(environments))
        width = 0.2
        
        for i, algo in enumerate(algorithms):
            offset = width * (i - len(algorithms) / 2 + 0.5)
            color = self.colors.get(algo, '#000000')
            
            bars = ax.bar(
                x + offset,
                data_matrix[i],
                width,
                label=algo,
                color=color,
                alpha=0.8,
                yerr=std_matrix[i],
                capsize=5,
                error_kw={'linewidth': 1.5, 'alpha': 0.7}
            )
        
        ax.set_xlabel('Environment', fontweight='bold', fontsize=14)
        ax.set_ylabel('Final Performance (Mean Reward)', fontweight='bold', fontsize=14)
        ax.set_title('Final Performance Comparison Across Environments', 
                    fontweight='bold', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels([e.replace('_', '\n') for e in environments], rotation=0)
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        
        if save:
            filename = self.output_dir / "final_performance_comparison.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot: {filename}")
        
        return fig
    
    def plot_convergence_speed(
        self,
        metrics: Dict[str, Dict[str, Dict]],
        save: bool = True
    ) -> plt.Figure:
        """
        Plot convergence speed comparison
        
        Args:
            metrics: {algorithm: {environment: analysis_results}}
            save: Whether to save
            
        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Organize data
        environments = set()
        for algo in metrics:
            environments.update(metrics[algo].keys())
        environments = sorted(list(environments))
        algorithms = sorted(metrics.keys())
        
        # Data for plot 1: Convergence episode
        conv_episodes = np.zeros((len(algorithms), len(environments)))
        for i, algo in enumerate(algorithms):
            for j, env in enumerate(environments):
                if env in metrics[algo]:
                    ce = metrics[algo][env]['convergence']['convergence_episode']
                    conv_episodes[i, j] = ce if ce is not None else np.nan
                else:
                    conv_episodes[i, j] = np.nan
        
        # Plot 1: Convergence episodes (grouped bar)
        x = np.arange(len(environments))
        width = 0.2
        
        for i, algo in enumerate(algorithms):
            offset = width * (i - len(algorithms) / 2 + 0.5)
            color = self.colors.get(algo, '#000000')
            
            axes[0].bar(
                x + offset,
                conv_episodes[i],
                width,
                label=algo,
                color=color,
                alpha=0.8
            )
        
        axes[0].set_xlabel('Environment', fontweight='bold')
        axes[0].set_ylabel('Episodes to Convergence', fontweight='bold')
        axes[0].set_title('Convergence Speed Comparison', fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([e.replace('_', '\n') for e in environments], rotation=0)
        axes[0].legend(loc='best', framealpha=0.9)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Convergence rate (scatter)
        for algo in algorithms:
            conv_rates = []
            max_rewards = []
            
            for env in environments:
                if env in metrics[algo]:
                    conv = metrics[algo][env]['convergence']
                    conv_rates.append(conv['convergence_rate'])
                    max_rewards.append(conv['max_reward'])
            
            color = self.colors.get(algo, '#000000')
            axes[1].scatter(conv_rates, max_rewards, label=algo, 
                          color=color, s=100, alpha=0.7, edgecolors='black')
        
        axes[1].set_xlabel('Convergence Rate (Reward/Episode)', fontweight='bold')
        axes[1].set_ylabel('Maximum Reward Achieved', fontweight='bold')
        axes[1].set_title('Convergence Rate vs Max Reward', fontweight='bold')
        axes[1].legend(loc='best', framealpha=0.9)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filename = self.output_dir / "convergence_speed_comparison.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot: {filename}")
        
        return fig
    
    def plot_training_stability(self, metrics: Dict[str, Dict[str, Dict]], save: bool = True) -> plt.Figure:
        """
        Plot training stability comparison
        
        Args:
            metrics: {algorithm: {environment: analysis_results}}
            save: Whether to save
            
        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        # Organize data
        environments = set()
        for algo in metrics:
            environments.update(metrics[algo].keys())
        environments = sorted(list(environments))
        algorithms = sorted(metrics.keys())
        
        # Metric 1: Coefficient of Variation
        cv_data = np.zeros((len(algorithms), len(environments)))
        for i, algo in enumerate(algorithms):
            for j, env in enumerate(environments):
                if env in metrics[algo]:
                    cv_data[i, j] = metrics[algo][env]['stability']['coefficient_of_variation']
                else:
                    cv_data[i, j] = np.nan
        
        x = np.arange(len(environments))
        width = 0.2
        for i, algo in enumerate(algorithms):
            offset = width * (i - len(algorithms) / 2 + 0.5)
            color = self.colors.get(algo, '#000000')
            axes[0].bar(x + offset, cv_data[i], width, label=algo, color=color, alpha=0.8)
        
        axes[0].set_xlabel('Environment', fontweight='bold')
        axes[0].set_ylabel('Coefficient of Variation', fontweight='bold')
        axes[0].set_title('Training Stability (CV - Lower is Better)', fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([e.replace('_', '\n') for e in environments], rotation=0, fontsize=9)
        axes[0].legend(loc='best', framealpha=0.9)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Metric 2: Volatility
        volatility_data = np.zeros((len(algorithms), len(environments)))
        for i, algo in enumerate(algorithms):
            for j, env in enumerate(environments):
                if env in metrics[algo]:
                    volatility_data[i, j] = metrics[algo][env]['stability']['volatility']
                else:
                    volatility_data[i, j] = np.nan
        
        for i, algo in enumerate(algorithms):
            offset = width * (i - len(algorithms) / 2 + 0.5)
            color = self.colors.get(algo, '#000000')
            axes[1].bar(x + offset, volatility_data[i], width, label=algo, color=color, alpha=0.8)
        
        axes[1].set_xlabel('Environment', fontweight='bold')
        axes[1].set_ylabel('Volatility', fontweight='bold')
        axes[1].set_title('Reward Volatility (Lower is Better)', fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([e.replace('_', '\n') for e in environments], rotation=0, fontsize=9)
        axes[1].legend(loc='best', framealpha=0.9)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Metric 3: Final Std
        final_std_data = np.zeros((len(algorithms), len(environments)))
        for i, algo in enumerate(algorithms):
            for j, env in enumerate(environments):
                if env in metrics[algo]:
                    final_std_data[i, j] = metrics[algo][env]['stability']['final_std']
                else:
                    final_std_data[i, j] = np.nan
        
        for i, algo in enumerate(algorithms):
            offset = width * (i - len(algorithms) / 2 + 0.5)
            color = self.colors.get(algo, '#000000')
            axes[2].bar(x + offset, final_std_data[i], width, label=algo, color=color, alpha=0.8)
        
        axes[2].set_xlabel('Environment', fontweight='bold')
        axes[2].set_ylabel('Final Std Deviation', fontweight='bold')
        axes[2].set_title('Final Phase Stability (Lower is Better)', fontweight='bold')
        axes[2].set_xticks(x)
        axes[2].set_xticklabels([e.replace('_', '\n') for e in environments], rotation=0, fontsize=9)
        axes[2].legend(loc='best', framealpha=0.9)
        axes[2].grid(True, alpha=0.3, axis='y')
        
        # Metric 4: Radar chart for one environment (average)
        # Select environment with most data
        env_counts = {env: sum(1 for algo in algorithms if env in metrics[algo]) for env in environments}
        best_env = max(env_counts, key=env_counts.get)
        
        # Prepare radar chart data
        categories = ['Final\nPerformance', 'Stability\n(1-CV)', 'Low\nVolatility', 'Convergence\nSpeed']
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        ax = plt.subplot(224, projection='polar')
        
        for algo in algorithms:
            if best_env not in metrics[algo]:
                continue
            
            m = metrics[algo][best_env]
            
            # Normalize metrics to [0, 1] (higher is better)
            final_perf = m['final_performance']['mean']
            stability = 1.0 / (1.0 + m['stability']['coefficient_of_variation'])
            low_volatility = 1.0 - min(1.0, m['stability']['volatility'])
            conv_speed = 1.0 / (1.0 + m['convergence']['convergence_episode'] / 1000.0) if m['convergence']['convergence_episode'] else 0.5
            
            # Normalize final_perf to [0, 1] based on data range
            all_perf = [metrics[a][best_env]['final_performance']['mean'] for a in algorithms if best_env in metrics[a]]
            min_perf, max_perf = min(all_perf), max(all_perf)
            if max_perf > min_perf:
                final_perf_norm = (final_perf - min_perf) / (max_perf - min_perf)
            else:
                final_perf_norm = 0.5
            
            values = [final_perf_norm, stability, low_volatility, conv_speed]
            values += values[:1]  # Complete the circle
            
            color = self.colors.get(algo, '#000000')
            ax.plot(angles, values, 'o-', linewidth=2, label=algo, color=color)
            ax.fill(angles, values, alpha=0.15, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_title(f'Overall Performance Radar\n({best_env.replace("_", " ").title()})', 
                    fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filename = self.output_dir / "training_stability_comparison.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot: {filename}")
        
        return fig
    
    def plot_sample_efficiency(self, metrics: Dict[str, Dict[str, Dict]], save: bool = True) -> plt.Figure:
        """
        Plot sample efficiency comparison
        
        Args:
            metrics: {algorithm: {environment: analysis_results}}
            save: Whether to save
            
        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Organize data
        environments = set()
        for algo in metrics:
            environments.update(metrics[algo].keys())
        environments = sorted(list(environments))
        algorithms = sorted(metrics.keys())
        
        # Plot 1: Episodes to thresholds
        thresholds = ['25%', '50%', '75%', '90%']
        threshold_data = {thresh: np.zeros((len(algorithms), len(environments))) for thresh in thresholds}
        
        for i, algo in enumerate(algorithms):
            for j, env in enumerate(environments):
                if env in metrics[algo]:
                    eff = metrics[algo][env]['sample_efficiency']['episodes_to_threshold']
                    for thresh in thresholds:
                        val = eff.get(thresh)
                        threshold_data[thresh][i, j] = val if val is not None else np.nan
        
        x = np.arange(len(algorithms))
        width = 0.2
        
        # Select one representative environment
        env_idx = 0  # First environment
        env_name = environments[env_idx]
        
        for i, thresh in enumerate(thresholds):
            offset = width * (i - len(thresholds) / 2 + 0.5)
            data = threshold_data[thresh][:, env_idx]
            
            axes[0].bar(x + offset, data, width, label=f'{thresh} of Max', alpha=0.8)
        
        axes[0].set_xlabel('Algorithm', fontweight='bold')
        axes[0].set_ylabel('Episodes Required', fontweight='bold')
        axes[0].set_title(f'Sample Efficiency - {env_name.replace("_", " ").title()}', fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(algorithms)
        axes[0].legend(loc='best', framealpha=0.9)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Average reward progression
        for algo in algorithms:
            if env_name in metrics[algo]:
                reward_per_100 = metrics[algo][env_name]['sample_efficiency']['reward_per_100_episodes']
                episodes = np.arange(len(reward_per_100)) * 100
                
                color = self.colors.get(algo, '#000000')
                linestyle = self.linestyles.get(algo, '-')
                axes[1].plot(episodes, reward_per_100, label=algo, 
                           color=color, linestyle=linestyle, linewidth=2.5)
        
        axes[1].set_xlabel('Episodes', fontweight='bold')
        axes[1].set_ylabel('Average Reward (per 100 episodes)', fontweight='bold')
        axes[1].set_title(f'Reward Progression - {env_name.replace("_", " ").title()}', fontweight='bold')
        axes[1].legend(loc='best', framealpha=0.9)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filename = self.output_dir / "sample_efficiency_comparison.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot: {filename}")
        
        return fig
    
    def plot_environment_adaptability(self, metrics: Dict[str, Dict[str, Dict]], save: bool = True) -> plt.Figure:
        """
        Plot algorithm adaptability across environments
        
        Args:
            metrics: {algorithm: {environment: analysis_results}}
            save: Whether to save
            
        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Organize data
        environments = set()
        for algo in metrics:
            environments.update(metrics[algo].keys())
        environments = sorted(list(environments))
        algorithms = sorted(metrics.keys())
        
        # Metric 1: Normalized final performance across environments
        perf_matrix = np.zeros((len(algorithms), len(environments)))
        for i, algo in enumerate(algorithms):
            for j, env in enumerate(environments):
                if env in metrics[algo]:
                    perf_matrix[i, j] = metrics[algo][env]['final_performance']['mean']
                else:
                    perf_matrix[i, j] = np.nan
        
        # Normalize per environment (column-wise)
        norm_perf_matrix = np.zeros_like(perf_matrix)
        for j in range(len(environments)):
            col = perf_matrix[:, j]
            valid_col = col[~np.isnan(col)]
            if len(valid_col) > 0:
                min_val, max_val = np.min(valid_col), np.max(valid_col)
                if max_val > min_val:
                    norm_perf_matrix[:, j] = (col - min_val) / (max_val - min_val)
                else:
                    norm_perf_matrix[:, j] = 0.5
        
        # Heatmap
        im = axes[0].imshow(norm_perf_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        axes[0].set_xticks(np.arange(len(environments)))
        axes[0].set_yticks(np.arange(len(algorithms)))
        axes[0].set_xticklabels([e.replace('_', '\n') for e in environments], fontsize=9)
        axes[0].set_yticklabels(algorithms)
        
        # Add text annotations
        for i in range(len(algorithms)):
            for j in range(len(environments)):
                if not np.isnan(norm_perf_matrix[i, j]):
                    text = axes[0].text(j, i, f'{norm_perf_matrix[i, j]:.2f}',
                                      ha="center", va="center", color="black", fontsize=9)
        
        axes[0].set_title('Normalized Performance Heatmap\n(Higher is Better)', fontweight='bold')
        fig.colorbar(im, ax=axes[0], label='Normalized Performance')
        
        # Metric 2: Performance variance across environments (adaptability)
        algo_means = []
        algo_stds = []
        
        for algo in algorithms:
            perfs = [metrics[algo][env]['final_performance']['mean'] 
                    for env in environments if env in metrics[algo]]
            if perfs:
                algo_means.append(np.mean(perfs))
                algo_stds.append(np.std(perfs))
            else:
                algo_means.append(0)
                algo_stds.append(0)
        
        colors_list = [self.colors.get(algo, '#000000') for algo in algorithms]
        axes[1].bar(algorithms, algo_means, yerr=algo_stds, 
                   color=colors_list, alpha=0.8, capsize=5,
                   error_kw={'linewidth': 1.5})
        
        axes[1].set_xlabel('Algorithm', fontweight='bold')
        axes[1].set_ylabel('Average Final Performance', fontweight='bold')
        axes[1].set_title('Cross-Environment Performance\n(Error bars = Std across environments)', fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        
        if save:
            filename = self.output_dir / "environment_adaptability.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot: {filename}")
        
        return fig
