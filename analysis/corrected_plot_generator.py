"""
Corrected Plot Generator for MARL Training Analysis
Fixed issues with title positioning, learning curves design, and performance comparison normalization
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import colorsys
from scipy import stats

logger = logging.getLogger(__name__)

# Enhanced publication-quality style settings with proper title spacing
plt.rcParams.update({
    'font.size': 16,
    'font.family': 'DejaVu Sans',
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'axes.titlepad': 25,  # Increased title padding
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 15,
    'figure.titlesize': 24,
    'figure.titleweight': 'bold',
    'figure.dpi': 120,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    'savefig.pad_inches': 0.3,  # Add padding around saved figure
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': ':',
    'grid.linewidth': 0.8,
    'lines.linewidth': 3,
    'lines.markersize': 8,
    'axes.facecolor': '#FAFAFA',
    'figure.facecolor': 'white',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.edgecolor': '#333333',
    'axes.linewidth': 1.2,
    'legend.framealpha': 0.95,
    'legend.fancybox': True,
    'legend.shadow': True,
})


class CorrectedPlotGenerator:
    """Corrected plot generator with fixed issues"""

    def __init__(self, output_dir: str = "corrected_plots"):
        """
        Initialize corrected plot generator

        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Professional color palette with better contrast
        self.colors = {
            'QMIX': '#1f77b4',    # Professional Blue
            'IQL': '#ff7f0e',     # Orange
            'VDN': '#2ca02c',     # Green
            'COMA': '#d62728',    # Red
            'MADDPG': '#9467bd',  # Purple
            'MAPPO': '#8c564b',   # Brown
            'Default': '#17becf'   # Cyan
        }

        # Enhanced line styles and markers for better distinction
        self.line_styles = {
            'QMIX': {'style': '-', 'marker': 'o', 'alpha': 0.9},
            'IQL': {'style': '--', 'marker': 's', 'alpha': 0.9},
            'VDN': {'style': '-.', 'marker': '^', 'alpha': 0.9},
            'COMA': {'style': ':', 'marker': 'D', 'alpha': 0.9},
            'MADDPG': {'style': (0, (3, 1, 1, 1)), 'marker': 'v', 'alpha': 0.9},
            'MAPPO': {'style': (0, (5, 1, 2, 1)), 'marker': 'p', 'alpha': 0.9}
        }

        # Generate additional colors for more algorithms
        self._generate_additional_colors()

    def _generate_additional_colors(self):
        """Generate additional distinguishable colors using color theory"""
        base_hues = [0, 30, 60, 120, 180, 240, 270, 300]  # Red, orange, yellow, green, cyan, blue, purple, magenta

        additional_algos = ['MADDPG', 'MAPPO', 'QMIX_STAR', 'VDN_SHARED', 'IQL_REPLAY']

        for i, algo in enumerate(additional_algos):
            if algo not in self.colors:
                hue = base_hues[i % len(base_hues)] / 360
                # Use different saturation and lightness for variety
                saturation = 0.7 + (i // len(base_hues)) * 0.1
                lightness = 0.4 + (i // len(base_hues)) * 0.1
                rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
                color = '#{:02x}{:02x}{:02x}'.format(
                    int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
                )
                self.colors[algo] = color

    def calculate_running_statistics(self, rewards: np.ndarray, window: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate running mean and confidence intervals

        Args:
            rewards: Array of episode rewards
            window: Window size for running statistics

        Returns:
            Tuple of (episodes, running_mean, confidence_interval)
        """
        if len(rewards) < window:
            window = len(rewards) // 2 if len(rewards) > 1 else 1

        running_means = []
        confidence_intervals = []

        for i in range(window, len(rewards) + 1):
            window_data = rewards[i-window:i]
            running_mean = np.mean(window_data)
            running_means.append(running_mean)

            # Calculate 95% confidence interval
            if len(window_data) > 1:
                sem = stats.sem(window_data)  # Standard error of mean
                ci_95 = stats.t.interval(0.95, len(window_data)-1, loc=running_mean, scale=sem)
                confidence_intervals.append((ci_95[1] - ci_95[0]) / 2)  # Half-width
            else:
                confidence_intervals.append(0)

        episodes = np.arange(window, len(rewards) + 1)
        running_means = np.array(running_means)
        confidence_intervals = np.array(confidence_intervals)

        return episodes, running_means, confidence_intervals

    def plot_corrected_learning_curves(
        self,
        data: Dict[str, Dict[str, List[float]]],
        environment: str,
        window: int = 100,
        save: bool = True
    ) -> plt.Figure:
        """
        Plot corrected learning curves with running statistics instead of raw data

        Args:
            data: {algorithm: {env: rewards}}
            environment: Environment name
            window: Window size for running statistics
            save: Whether to save the figure

        Returns:
            matplotlib Figure
        """
        # Create larger figure with proper spacing
        fig, ax = plt.subplots(figsize=(24, 14))

        available_algos = [algo for algo in data if environment in data[algo]]

        for algo in available_algos:
            rewards = np.array(data[algo][environment])

            # Calculate running statistics
            episodes, running_means, confidence_intervals = self.calculate_running_statistics(
                rewards, window
            )

            color = self.colors.get(algo, self.colors['Default'])
            line_config = self.line_styles.get(algo, {'style': '-', 'marker': 'o', 'alpha': 0.9})

            # Plot running mean with confidence intervals
            ax.plot(episodes, running_means,
                    label=algo, color=color,
                    linestyle=line_config['style'],
                    linewidth=4,
                    marker=line_config['marker'],
                    markersize=6,
                    markevery=len(episodes)//25,  # Show markers occasionally
                    alpha=line_config['alpha'])

            # Add confidence interval band
            ax.fill_between(
                episodes,
                running_means - confidence_intervals,
                running_means + confidence_intervals,
                alpha=0.25,
                color=color,
                step='mid',
                label=f'{algo} (95% CI)' if algo == available_algos[0] else None
            )

            # Note: Stability can be observed from the width of confidence intervals

        # Enhanced styling with proper title positioning
        env_title = environment.replace('_', ' ').title()

        ax.set_xlabel('Training Episode', fontweight='bold', fontsize=18)
        ax.set_ylabel('Running Mean Reward', fontweight='bold', fontsize=18)
        ax.set_title(f'Learning Progress Analysis - {env_title}\n'
                    f'Running Mean with {window}-Episode Window and 95% Confidence Intervals',
                    fontweight='bold', fontsize=22, pad=35)  # Increased padding

        # Enhanced legend with better positioning
        legend = ax.legend(loc='best', fontsize=14, framealpha=0.95,
                          fancybox=True, shadow=True, ncol=1)
        legend.get_frame().set_facecolor('white')

        ax.grid(True, alpha=0.4, linestyle=':', linewidth=1.2)
        ax.set_facecolor('#FAFAFA')

        # Add convergence indicator lines
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Zero Line')

        # Adjust layout to prevent title cutoff
        plt.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.92)

        if save:
            filename = self.output_dir / f"corrected_learning_curves_{environment}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none',
                       pad_inches=0.5)  # Add padding around the saved figure
            logger.info(f"Saved corrected plot: {filename}")

        return fig

    def plot_normalized_performance_comparison(
        self,
        metrics: Dict[str, Dict[str, Dict]],
        save: bool = True
    ) -> plt.Figure:
        """
        Create normalized performance comparison with relative comparison within each environment

        Args:
            metrics: {algorithm: {environment: analysis_results}}
            save: Whether to save the figure

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

        # Normalize per environment (column-wise normalization)
        normalized_matrix = np.zeros_like(data_matrix)
        normalized_std = np.zeros_like(std_matrix)

        for j, env in enumerate(environments):
            col = data_matrix[:, j]
            valid_indices = ~np.isnan(col)

            if np.sum(valid_indices) > 1:
                valid_col = col[valid_indices]
                min_val = np.min(valid_col)
                max_val = np.max(valid_col)

                if max_val > min_val:
                    # Normalize to [0, 1] where 1 is best performance
                    normalized_matrix[valid_indices, j] = (valid_col - min_val) / (max_val - min_val)

                    # Also normalize standard deviation proportionally
                    for i in range(len(algorithms)):
                        if valid_indices[i]:
                            range_val = max_val - min_val
                            if range_val > 0:
                                normalized_std[i, j] = std_matrix[i, j] / range_val
                else:
                    # All values are the same
                    normalized_matrix[valid_indices, j] = 0.5
                    normalized_std[valid_indices, j] = 0

        # Create large figure with proper spacing
        fig, ax = plt.subplots(figsize=(22, 14))

        x = np.arange(len(environments))
        width = 0.15  # Adjusted width for better visibility

        # Create bar plot with enhanced styling
        bars = []
        for i, algo in enumerate(algorithms):
            offset = width * (i - len(algorithms) / 2 + 0.5)
            color = self.colors.get(algo, self.colors['Default'])

            bar = ax.bar(
                x + offset,
                normalized_matrix[i],
                width,
                label=algo,
                color=color,
                alpha=0.85,
                edgecolor='black',
                linewidth=1.5,
                yerr=normalized_std[i] * 0.5,  # Scaled error bars
                capsize=6,
                error_kw={'linewidth': 2, 'alpha': 0.8, 'color': 'black'}
            )
            bars.append(bar)

            # Add value labels on bars (show normalized values)
            for j, (bar_rect, value) in enumerate(zip(bar.patches, normalized_matrix[i])):
                if not np.isnan(value):
                    height = bar_rect.get_height()
                    ax.text(bar_rect.get_x() + bar_rect.get_width()/2.,
                           height + 0.02,  # Adjusted position
                           f'{value:.2f}',
                           ha='center', va='bottom',
                           fontsize=11, fontweight='bold')

        # Enhanced styling with proper title positioning
        ax.set_xlabel('Environment', fontweight='bold', fontsize=18)
        ax.set_ylabel('Normalized Performance Score (0=worst, 1=best within environment)',
                     fontweight='bold', fontsize=18)
        ax.set_title('Relative Performance Comparison Across Environments\n'
                    '(Each environment normalized independently - shows relative ranking within each environment)',
                    fontweight='bold', fontsize=22, pad=35)

        ax.set_xticks(x)
        ax.set_xticklabels([e.replace('_', ' ').title() for e in environments],
                          rotation=45, ha='right', fontsize=14)

        # Enhanced legend with better positioning
        legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.08),
                         ncol=len(algorithms), framealpha=0.95,
                         fancybox=True, shadow=True, fontsize=15)
        legend.get_frame().set_facecolor('white')

        ax.grid(True, alpha=0.4, axis='y', linestyle=':', linewidth=1.2)
        ax.set_facecolor('#FAFAFA')
        ax.set_ylim(0, 1.1)  # Fixed y-axis for normalized scores

        # Add reference lines
        ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Average Performance')
        ax.axhline(y=1.0, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Best Performance')

        # Adjust layout to prevent title cutoff
        plt.subplots_adjust(top=0.88, bottom=0.12, left=0.08, right=0.95)

        if save:
            filename = self.output_dir / "corrected_normalized_performance_comparison.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none',
                       pad_inches=0.5)
            logger.info(f"Saved corrected normalized comparison: {filename}")

        return fig

    def create_enhanced_performance_heatmap(
        self,
        metrics: Dict[str, Dict[str, Dict]],
        save: bool = True
    ) -> plt.Figure:
        """
        Create an enhanced heatmap showing performance patterns

        Args:
            metrics: {algorithm: {environment: analysis_results}}
            save: Whether to save the figure

        Returns:
            matplotlib Figure
        """
        # Organize data
        environments = set()
        for algo in metrics:
            environments.update(metrics[algo].keys())
        environments = sorted(list(environments))

        algorithms = sorted(metrics.keys())

        # Create performance matrix
        performance_matrix = np.zeros((len(algorithms), len(environments)))

        for i, algo in enumerate(algorithms):
            for j, env in enumerate(environments):
                if env in metrics[algo]:
                    performance_matrix[i, j] = metrics[algo][env]['final_performance']['mean']
                else:
                    performance_matrix[i, j] = np.nan

        # Create large figure
        fig, ax = plt.subplots(figsize=(16, 12))

        # Create heatmap with enhanced styling
        im = ax.imshow(performance_matrix, cmap='RdYlGn', aspect='auto',
                      interpolation='nearest')

        # Set ticks and labels
        ax.set_xticks(np.arange(len(environments)))
        ax.set_yticks(np.arange(len(algorithms)))
        ax.set_xticklabels([e.replace('_', ' ').title() for e in environments],
                          rotation=45, ha='right', fontsize=14)
        ax.set_yticklabels(algorithms, fontsize=14)

        # Add text annotations with performance values and rankings
        for i in range(len(algorithms)):
            for j in range(len(environments)):
                if not np.isnan(performance_matrix[i, j]):
                    value = performance_matrix[i, j]

                    # Calculate rank within this environment
                    column_data = performance_matrix[:, j]
                    valid_data = column_data[~np.isnan(column_data)]
                    if len(valid_data) > 0:
                        sorted_data = np.sort(valid_data)
                        rank = np.sum(sorted_data < value) + 1
                        total = len(valid_data)

                        text = f'{value:.1f}\n({rank}/{total})'
                        color = 'white' if value < np.nanmean(valid_data) else 'black'
                    else:
                        text = f'{value:.1f}'
                        color = 'black'

                    ax.text(j, i, text, ha="center", va="center", color=color,
                           fontsize=11, fontweight='bold')

        ax.set_title('Algorithm Performance Heatmap with Rankings\n'
                    '(Values show performance, numbers show rank within each environment)',
                    fontweight='bold', fontsize=20, pad=30)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, label='Final Performance Score')
        cbar.ax.tick_params(labelsize=12)

        plt.tight_layout()

        if save:
            filename = self.output_dir / "enhanced_performance_heatmap.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            logger.info(f"Saved enhanced heatmap: {filename}")

        return fig

    def plot_performance_distribution_analysis(
        self,
        metrics: Dict[str, Dict[str, Dict]],
        save: bool = True
    ) -> plt.Figure:
        """
        Create distribution analysis showing performance consistency across environments

        Args:
            metrics: {algorithm: {environment: analysis_results}}
            save: Whether to save the figure

        Returns:
            matplotlib Figure
        """
        algorithms = sorted(metrics.keys())

        # Calculate statistics for each algorithm across environments
        performance_stats = {}
        for algo in algorithms:
            performances = []
            for env in metrics[algo]:
                performances.append(metrics[algo][env]['final_performance']['mean'])

            performance_stats[algo] = {
                'mean': np.mean(performances),
                'std': np.std(performances),
                'min': np.min(performances),
                'max': np.max(performances),
                'values': performances
            }

        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Algorithm Performance Distribution Analysis\n'
                    'Consistency and Reliability Across Environments',
                    fontsize=22, fontweight='bold')

        # 1. Box plot for performance distribution
        ax1.boxplot([performance_stats[algo]['values'] for algo in algorithms],
                   labels=algorithms, patch_artist=True)

        colors = [self.colors.get(algo, '#000000') for algo in algorithms]
        for patch, color in zip(ax1.artists, colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax1.set_title('Performance Distribution (Box Plot)', fontweight='bold', fontsize=16)
        ax1.set_ylabel('Final Performance', fontweight='bold', fontsize=14)
        ax1.grid(True, alpha=0.3)

        # 2. Mean performance with error bars
        means = [performance_stats[algo]['mean'] for algo in algorithms]
        stds = [performance_stats[algo]['std'] for algo in algorithms]

        bars = ax2.bar(algorithms, means, yerr=stds, color=colors,
                      alpha=0.8, capsize=8, edgecolor='black', linewidth=1.5)

        # Add value labels
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2.,
                    height + std + max(means)*0.02,
                    f'{mean:.1f}±{std:.1f}', ha='center', va='bottom',
                    fontweight='bold', fontsize=11)

        ax2.set_title('Mean Performance with Standard Deviation', fontweight='bold', fontsize=16)
        ax2.set_ylabel('Final Performance', fontweight='bold', fontsize=14)
        ax2.grid(True, alpha=0.3, axis='y')

        # 3. Coefficient of variation (consistency measure)
        cvs = [performance_stats[algo]['std'] / abs(performance_stats[algo]['mean']) if performance_stats[algo]['mean'] != 0 else np.inf
               for algo in algorithms]

        bars3 = ax3.bar(algorithms, cvs, color=colors, alpha=0.8,
                        edgecolor='black', linewidth=1.5)

        # Add value labels
        for bar, cv in zip(bars3, cvs):
            height = bar.get_height()
            if np.isfinite(height):
                ax3.text(bar.get_x() + bar.get_width()/2., height + max(cvs)*0.02,
                        f'{cv:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

        ax3.set_title('Coefficient of Variation (Lower = More Consistent)', fontweight='bold', fontsize=16)
        ax3.set_ylabel('CV (Std/Mean)', fontweight='bold', fontsize=14)
        ax3.grid(True, alpha=0.3, axis='y')

        # 4. Performance range (max - min)
        ranges = [performance_stats[algo]['max'] - performance_stats[algo]['min']
                  for algo in algorithms]

        bars4 = ax4.bar(algorithms, ranges, color=colors, alpha=0.8,
                        edgecolor='black', linewidth=1.5)

        # Add value labels
        for bar, range_val in zip(bars4, ranges):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + max(ranges)*0.02,
                    f'{range_val:.1f}', ha='center', va='bottom',
                    fontweight='bold', fontsize=11)

        ax4.set_title('Performance Range (Max - Min)', fontweight='bold', fontsize=16)
        ax4.set_ylabel('Performance Range', fontweight='bold', fontsize=14)
        ax4.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save:
            filename = self.output_dir / "performance_distribution_analysis.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            logger.info(f"Saved distribution analysis: {filename}")

        return fig


if __name__ == "__main__":
    # Test the corrected plot generator
    logging.basicConfig(level=logging.INFO)

    # Create sample data for testing
    sample_data = {
        'QMIX': {
            'CM_hard': np.random.randn(2000).cumsum() + np.random.randn(2000) * 10,
            'DEM_hard': np.random.randn(2000).cumsum() + np.random.randn(2000) * 15,
        },
        'IQL': {
            'CM_hard': np.random.randn(2000).cumsum() + np.random.randn(2000) * 8,
            'DEM_hard': np.random.randn(2000).cumsum() + np.random.randn(2000) * 12,
        }
    }

    generator = CorrectedPlotGenerator("test_corrected_plots")

    # Test corrected learning curves
    fig = generator.plot_corrected_learning_curves(sample_data, 'CM_hard')
    plt.show()

    print("Corrected plot generator test complete!")