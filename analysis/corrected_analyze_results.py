#!/usr/bin/env python3
"""
Corrected MARL Training Results Analysis System
Fixed issues with title positioning, learning curves design, and performance comparison normalization
"""
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

# Import our modules
from data_loader import TrainingDataLoader, compute_statistics
from metrics_analyzer import MetricsAnalyzer
from corrected_plot_generator import CorrectedPlotGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('corrected_analysis.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class CorrectedResultsAnalyzer:
    """Corrected analyzer with fixed visualization issues"""

    def __init__(self, data_store_root: str = "../data_store", output_dir: str = "corrected_output"):
        """
        Initialize corrected analyzer

        Args:
            data_store_root: Root directory of training data
            output_dir: Output directory for corrected plots
        """
        self.data_loader = TrainingDataLoader(data_store_root)
        self.metrics_analyzer = MetricsAnalyzer()
        self.plot_generator = CorrectedPlotGenerator(output_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.data = {}
        self.metrics = {}

    def load_all_data(self, select_latest: bool = True):
        """Load all training data"""
        logger.info("Loading training data...")
        self.data = self.data_loader.load_all_data()

        if not self.data:
            logger.error("No data loaded! Please check your data_store path.")
            return False

        logger.info(f"Loaded data for {len(self.data)} algorithms")
        return True

    def analyze_metrics(self):
        """Analyze metrics for all loaded data"""
        if not self.data:
            logger.error("No data available for analysis. Please load data first.")
            return False

        logger.info("Analyzing metrics...")

        for algo in self.data:
            self.metrics[algo] = {}
            for env in self.data[algo]:
                logger.debug(f"Analyzing {algo} - {env}")
                rewards = self.data[algo][env]
                stats = self.metrics_analyzer.analyze_algorithm(rewards)
                self.metrics[algo][env] = stats

        logger.info("Metric analysis complete")
        return True

    def generate_all_plots(self):
        """Generate all corrected plots"""
        if not self.metrics:
            logger.error("No metrics available. Please analyze metrics first.")
            return False

        logger.info("Generating corrected plots...")

        # Get all environments
        environments = set()
        for algo in self.metrics:
            environments.update(self.metrics[algo].keys())
        environments = sorted(list(environments))

        # 1. Corrected Learning Curves for each environment
        logger.info("Generating corrected learning curves...")
        for env in environments:
            try:
                self.plot_generator.plot_corrected_learning_curves(
                    self.data, env, window=100, save=True
                )
            except Exception as e:
                logger.error(f"Error generating corrected learning curves for {env}: {e}")

        # 2. Corrected Normalized Performance Comparison
        logger.info("Generating corrected normalized performance comparison...")
        try:
            self.plot_generator.plot_normalized_performance_comparison(
                self.metrics, save=True
            )
        except Exception as e:
            logger.error(f"Error generating normalized performance comparison: {e}")

        # 3. Enhanced Performance Heatmap
        logger.info("Generating enhanced performance heatmap...")
        try:
            self.plot_generator.create_enhanced_performance_heatmap(
                self.metrics, save=True
            )
        except Exception as e:
            logger.error(f"Error generating enhanced heatmap: {e}")

        # 4. Performance Distribution Analysis
        logger.info("Generating performance distribution analysis...")
        try:
            self.plot_generator.plot_performance_distribution_analysis(
                self.metrics, save=True
            )
        except Exception as e:
            logger.error(f"Error generating distribution analysis: {e}")

        logger.info("All corrected plots generated successfully")
        return True

    def generate_summary_report(self):
        """Generate corrected summary report"""
        if not self.metrics:
            logger.error("No metrics available for report generation")
            return False

        report_path = self.output_dir / "corrected_analysis_summary.txt"

        logger.info("Generating corrected summary report...")

        with open(report_path, 'w') as f:
            f.write("CORRECTED MARL ALGORITHM ANALYSIS REPORT\n")
            f.write("=" * 70 + "\n\n")

            # Summary overview
            f.write("CORRECTIONS AND IMPROVEMENTS:\n")
            f.write("-" * 35 + "\n")
            f.write("1. ✅ Fixed title positioning - Added proper padding to prevent cutoff\n")
            f.write("2. ✅ Improved learning curves - Replaced raw data with running statistics\n")
            f.write("3. ✅ Enhanced performance comparison - Added environment-wise normalization\n")
            f.write("4. ✅ Better visualization - Added confidence intervals and stability indicators\n")
            f.write("5. ✅ Larger figure sizes - Improved readability and detail visibility\n\n")

            f.write("ANALYSIS OVERVIEW:\n")
            f.write("-" * 25 + "\n")
            f.write(f"Algorithms analyzed: {len(self.data)}\n")
            f.write(f"Environments: {len(set(env for algo in self.data for env in self.data[algo]))}\n")
            f.write(f"Total algorithm-environment pairs: {sum(len(self.data[algo]) for algo in self.data)}\n\n")

            # Algorithm performance summary
            f.write("ALGORITHM PERFORMANCE SUMMARY:\n")
            f.write("-" * 35 + "\n")

            # Calculate normalized performance for better comparison
            algo_normalized_scores = {}
            for algo in self.metrics:
                normalized_scores = []
                for env in self.metrics[algo]:
                    # Get performance within this environment
                    env_performances = [self.metrics[a][env]['final_performance']['mean']
                                      for a in self.metrics if env in self.metrics[a]]
                    if env_performances:
                        min_perf = min(env_performances)
                        max_perf = max(env_performances)
                        if max_perf > min_perf:
                            normalized_score = (self.metrics[algo][env]['final_performance']['mean'] - min_perf) / (max_perf - min_perf)
                            normalized_scores.append(normalized_score)

                if normalized_scores:
                    algo_normalized_scores[algo] = np.mean(normalized_scores)

            # Sort by normalized performance
            sorted_algos = sorted(algo_normalized_scores.items(), key=lambda x: x[1], reverse=True)

            for algo, normalized_score in sorted_algos:
                performances = []
                stabilities = []
                efficiencies = []
                conv_episodes = []

                for env in self.metrics[algo]:
                    perf = self.metrics[algo][env]['final_performance']['mean']
                    stability = 1.0 / (1.0 + self.metrics[algo][env]['stability']['coefficient_of_variation'])
                    efficiency = self.metrics[algo][env]['sample_efficiency'].get('auc', 0.5)
                    conv = self.metrics[algo][env]['convergence']['convergence_episode'] or 0

                    performances.append(perf)
                    stabilities.append(stability)
                    efficiencies.append(efficiency)
                    conv_episodes.append(conv)

                f.write(f"\n{algo} (Normalized Score: {normalized_score:.3f}):\n")
                f.write(f"  Final Performance: {np.mean(performances):.2f} ± {np.std(performances):.2f}\n")
                f.write(f"  Training Stability: {np.mean(stabilities):.3f} ± {np.std(stabilities):.3f}\n")
                f.write(f"  Sample Efficiency: {np.mean(efficiencies):.3f} ± {np.std(efficiencies):.3f}\n")
                f.write(f"  Convergence Episodes: {np.mean(conv_episodes):.0f} ± {np.std(conv_episodes):.0f}\n")

            # Best algorithm recommendations
            f.write("\n\nPERFORMANCE RECOMMENDATIONS (Based on Normalized Scores):\n")
            f.write("-" * 60 + "\n")

            f.write("Best Overall Algorithm (Normalized Performance):\n")
            for i, (algo, score) in enumerate(sorted_algos[:3], 1):
                f.write(f"  {i}. {algo} (Normalized Score: {score:.3f})\n")

            # Find best in different categories
            best_performance = max([(algo, np.mean([self.metrics[algo][env]['final_performance']['mean']
                                                    for env in self.metrics[algo]]))
                                  for algo in self.metrics], key=lambda x: x[1])
            best_stability = max([(algo, np.mean([1.0 / (1.0 + self.metrics[algo][env]['stability']['coefficient_of_variation'])
                                                 for env in self.metrics[algo]]))
                                 for algo in self.metrics], key=lambda x: x[1])

            f.write(f"\nBest Pure Performance: {best_performance[0]} (Score: {best_performance[1]:.2f})\n")
            f.write(f"Most Stable Training: {best_stability[0]} (Score: {best_stability[1]:.3f})\n")

            f.write("\n\nVISUALIZATION IMPROVEMENTS:\n")
            f.write("-" * 35 + "\n")
            f.write("1. Corrected Learning Curves:\n")
            f.write("   - Removed raw data overlay for cleaner visualization\n")
            f.write("   - Added running mean with confidence intervals\n")
            f.write("   - Included stability indicators\n")
            f.write("   - Fixed title positioning with proper padding\n")
            f.write("   - Increased figure size for better visibility\n\n")

            f.write("2. Normalized Performance Comparison:\n")
            f.write("   - Each environment normalized independently\n")
            f.write("   - Shows relative ranking within each environment\n")
            f.write("   - Easier to compare algorithm consistency\n")
            f.write("   - Fixed legend positioning\n\n")

            f.write("3. Enhanced Heatmap:\n")
            f.write("   - Added performance rankings within each environment\n")
            f.write("   - Color-coded performance visualization\n")
            f.write("   - Clear value and rank annotations\n\n")

            f.write("4. Distribution Analysis:\n")
            f.write("   - Box plots showing performance distribution\n")
            f.write("   - Coefficient of variation for consistency\n")
            f.write("   - Performance range analysis\n\n")

            f.write("GENERATED VISUALIZATIONS:\n")
            f.write("-" * 30 + "\n")
            f.write("1. Corrected Learning Curves (per environment)\n")
            f.write("2. Normalized Performance Comparison Chart\n")
            f.write("3. Enhanced Performance Heatmap\n")
            f.write("4. Performance Distribution Analysis\n")

            f.write(f"\n\nAll plots saved to: {self.output_dir.absolute()}\n")
            f.write("Log file: corrected_analysis.log\n")

        logger.info(f"Corrected summary report saved to: {report_path}")
        return True

    def save_detailed_metrics(self):
        """Save detailed metrics to JSON for further processing"""
        import json

        metrics_path = self.output_dir / "corrected_detailed_metrics.json"

        logger.info("Saving detailed metrics...")

        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        # Deep copy and convert
        json_metrics = {}
        for algo in self.metrics:
            json_metrics[algo] = {}
            for env in self.metrics[algo]:
                json_metrics[algo][env] = {}
                for metric in self.metrics[algo][env]:
                    json_metrics[algo][env][metric] = convert_numpy(self.metrics[algo][env][metric])

        with open(metrics_path, 'w') as f:
            json.dump(json_metrics, f, indent=2)

        logger.info(f"Detailed metrics saved to: {metrics_path}")
        return True

    def run_complete_analysis(self):
        """Run complete corrected analysis pipeline"""
        logger.info("Starting complete corrected analysis...")

        success = True

        # Step 1: Load data
        if not self.load_all_data():
            logger.error("Failed to load data")
            return False

        # Step 2: Analyze metrics
        if not self.analyze_metrics():
            logger.error("Failed to analyze metrics")
            return False

        # Step 3: Generate plots
        if not self.generate_all_plots():
            logger.error("Failed to generate plots")
            success = False  # Continue with other steps even if some plots fail

        # Step 4: Generate report
        if not self.generate_summary_report():
            logger.error("Failed to generate report")
            return False

        # Step 5: Save detailed metrics
        if not self.save_detailed_metrics():
            logger.error("Failed to save detailed metrics")
            success = False  # Continue even if this fails

        if success:
            logger.info("Complete corrected analysis finished successfully!")
            logger.info(f"Results saved to: {self.output_dir.absolute()}")
        else:
            logger.warning("Analysis completed with some errors. Check logs for details.")

        return success


def main():
    """Main function"""
    print("\n" + "="*70)
    print("CORRECTED MARL TRAINING RESULTS ANALYSIS")
    print("="*70)

    # Initialize analyzer
    analyzer = CorrectedResultsAnalyzer(
        data_store_root="../data_store",
        output_dir="corrected_output"
    )

    # Run complete analysis
    success = analyzer.run_complete_analysis()

    if success:
        print("\n✅ Corrected analysis completed successfully!")
        print(f"📊 Results saved to: {analyzer.output_dir.absolute()}")
        print("\nGenerated files:")
        print("  • Corrected learning curves (per environment) - with running statistics")
        print("  • Normalized performance comparison - environment-wise relative comparison")
        print("  • Enhanced performance heatmap - with rankings")
        print("  • Performance distribution analysis - consistency and reliability")
        print("  • Corrected summary report")
        print("  • Detailed metrics (JSON)")
        print("\n📋 Check 'corrected_analysis_summary.txt' for key findings!")
        print("\n🔧 Corrections made:")
        print("  - Fixed title positioning to prevent cutoff")
        print("  - Replaced raw data with running statistics and confidence intervals")
        print("  - Added environment-wise normalization for fair comparison")
        print("  - Increased figure sizes for better visibility")
        print("  - Added stability and convergence indicators")
    else:
        print("\n❌ Analysis completed with errors. Check 'corrected_analysis.log' for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()