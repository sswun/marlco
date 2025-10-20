"""
Metrics Analyzer for MARL Algorithms
Compute performance metrics: convergence speed, stability, sample efficiency, final performance
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.signal import find_peaks
import logging

logger = logging.getLogger(__name__)


class MetricsAnalyzer:
    """Analyze training metrics for MARL algorithms"""
    
    def __init__(self, convergence_threshold: float = 0.9, window_size: int = 100):
        """
        Initialize analyzer
        
        Args:
            convergence_threshold: Threshold ratio of final performance for convergence
            window_size: Window size for computing smoothed metrics
        """
        self.convergence_threshold = convergence_threshold
        self.window_size = window_size
    
    def smooth_curve(self, values: List[float], window: int = None) -> np.ndarray:
        """Apply moving average smoothing"""
        if window is None:
            window = self.window_size
        
        if len(values) < window:
            return np.array(values)
        
        return np.convolve(values, np.ones(window) / window, mode='valid')
    
    def compute_final_performance(
        self, 
        rewards: List[float], 
        portion: float = 0.1
    ) -> Dict:
        """
        Compute final performance (last portion of training)
        
        Args:
            rewards: Episode rewards
            portion: Portion of episodes to consider (default: last 10%)
            
        Returns:
            Dictionary with mean, std, median, min, max
        """
        n_episodes = max(1, int(len(rewards) * portion))
        final_rewards = np.array(rewards[-n_episodes:])
        
        return {
            'mean': float(np.mean(final_rewards)),
            'std': float(np.std(final_rewards)),
            'median': float(np.median(final_rewards)),
            'min': float(np.min(final_rewards)),
            'max': float(np.max(final_rewards)),
            'n_episodes': n_episodes
        }
    
    def compute_training_stability(self, rewards: List[float]) -> Dict:
        """
        Compute training stability metrics
        
        Args:
            rewards: Episode rewards
            
        Returns:
            Dictionary with stability metrics
        """
        rewards_array = np.array(rewards)
        
        # Coefficient of variation (lower is more stable)
        cv = np.std(rewards_array) / (np.abs(np.mean(rewards_array)) + 1e-8)
        
        # Variance of smoothed curve
        smoothed = self.smooth_curve(rewards)
        smoothed_var = np.var(smoothed) if len(smoothed) > 0 else 0
        
        # Count of sign changes in reward differences (volatility)
        diffs = np.diff(rewards_array)
        sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
        volatility = sign_changes / len(diffs) if len(diffs) > 0 else 0
        
        # Standard deviation of last 20% episodes (final stability)
        final_portion = max(1, len(rewards) // 5)
        final_std = np.std(rewards_array[-final_portion:])
        
        return {
            'coefficient_of_variation': float(cv),
            'smoothed_variance': float(smoothed_var),
            'volatility': float(volatility),  # Higher = more unstable
            'final_std': float(final_std),
            'overall_std': float(np.std(rewards_array))
        }
    
    def compute_convergence_speed(self, rewards: List[float]) -> Dict:
        """
        Compute convergence speed metrics
        
        Args:
            rewards: Episode rewards
            
        Returns:
            Dictionary with convergence metrics
        """
        smoothed = self.smooth_curve(rewards)
        
        if len(smoothed) == 0:
            return {
                'convergence_episode': None,
                'convergence_rate': 0.0,
                'auc': 0.0,
                'max_reward': 0.0
            }
        
        # Find maximum smoothed reward
        max_reward = np.max(smoothed)
        threshold = max_reward * self.convergence_threshold
        
        # Find first episode where smoothed reward exceeds threshold
        convergence_episode = None
        converged_indices = np.where(smoothed >= threshold)[0]
        
        if len(converged_indices) > 0:
            convergence_episode = int(converged_indices[0])
        
        # Convergence rate: slope of linear fit in early phase
        early_phase = min(len(smoothed), 500)
        if early_phase > 10:
            x = np.arange(early_phase)
            y = smoothed[:early_phase]
            slope, _ = np.polyfit(x, y, 1)
            convergence_rate = float(slope)
        else:
            convergence_rate = 0.0
        
        # Area Under Curve (normalized)
        auc = float(np.trapz(smoothed) / len(smoothed)) if len(smoothed) > 0 else 0.0
        
        return {
            'convergence_episode': convergence_episode,
            'convergence_rate': convergence_rate,
            'auc': auc,
            'max_reward': float(max_reward),
            'threshold': float(threshold)
        }
    
    def compute_sample_efficiency(self, rewards: List[float]) -> Dict:
        """
        Compute sample efficiency metrics
        
        Args:
            rewards: Episode rewards
            
        Returns:
            Dictionary with sample efficiency metrics
        """
        smoothed = self.smooth_curve(rewards)
        
        if len(smoothed) == 0:
            return {
                'episodes_to_threshold': None,
                'reward_per_100_episodes': []
            }
        
        # Reward milestones
        max_reward = np.max(smoothed)
        thresholds = [0.25, 0.5, 0.75, 0.9]
        
        episodes_to_threshold = {}
        for thresh in thresholds:
            target = max_reward * thresh
            indices = np.where(smoothed >= target)[0]
            if len(indices) > 0:
                episodes_to_threshold[f'{int(thresh*100)}%'] = int(indices[0])
            else:
                episodes_to_threshold[f'{int(thresh*100)}%'] = None
        
        # Average reward every 100 episodes
        reward_per_100 = []
        for i in range(0, len(rewards), 100):
            chunk = rewards[i:i+100]
            if chunk:
                reward_per_100.append(float(np.mean(chunk)))
        
        return {
            'episodes_to_threshold': episodes_to_threshold,
            'reward_per_100_episodes': reward_per_100,
            'total_episodes': len(rewards)
        }
    
    def compute_learning_curve_features(self, rewards: List[float]) -> Dict:
        """
        Extract learning curve features
        
        Args:
            rewards: Episode rewards
            
        Returns:
            Dictionary with learning curve features
        """
        smoothed = self.smooth_curve(rewards)
        
        if len(smoothed) < 10:
            return {
                'n_peaks': 0,
                'n_valleys': 0,
                'monotonicity': 0.0,
                'trend': 'unknown'
            }
        
        # Find peaks and valleys
        peaks, _ = find_peaks(smoothed, distance=20)
        valleys, _ = find_peaks(-smoothed, distance=20)
        
        # Monotonicity: ratio of positive differences
        diffs = np.diff(smoothed)
        monotonicity = float(np.sum(diffs > 0) / len(diffs)) if len(diffs) > 0 else 0.0
        
        # Overall trend (linear regression slope)
        x = np.arange(len(smoothed))
        slope, _ = np.polyfit(x, smoothed, 1)
        
        if slope > 1.0:
            trend = 'strongly_increasing'
        elif slope > 0.1:
            trend = 'increasing'
        elif slope > -0.1:
            trend = 'stable'
        elif slope > -1.0:
            trend = 'decreasing'
        else:
            trend = 'strongly_decreasing'
        
        return {
            'n_peaks': len(peaks),
            'n_valleys': len(valleys),
            'monotonicity': monotonicity,
            'trend': trend,
            'slope': float(slope)
        }
    
    def analyze_algorithm(self, rewards: List[float]) -> Dict:
        """
        Comprehensive analysis of algorithm performance
        
        Args:
            rewards: Episode rewards
            
        Returns:
            Dictionary with all metrics
        """
        return {
            'final_performance': self.compute_final_performance(rewards),
            'stability': self.compute_training_stability(rewards),
            'convergence': self.compute_convergence_speed(rewards),
            'sample_efficiency': self.compute_sample_efficiency(rewards),
            'learning_curve': self.compute_learning_curve_features(rewards)
        }
    
    def compare_algorithms(
        self, 
        algorithms_data: Dict[str, List[float]]
    ) -> Dict[str, Dict]:
        """
        Compare multiple algorithms on same environment
        
        Args:
            algorithms_data: Dict of {algorithm_name: rewards}
            
        Returns:
            Comparison results for each algorithm
        """
        results = {}
        
        for algo_name, rewards in algorithms_data.items():
            results[algo_name] = self.analyze_algorithm(rewards)
        
        return results


if __name__ == "__main__":
    # Test with synthetic data
    logging.basicConfig(level=logging.INFO)
    
    # Create test data
    test_rewards = []
    for i in range(1000):
        # Simulated learning curve with noise
        base = 100 * (1 - np.exp(-i/200))
        noise = np.random.normal(0, 10)
        test_rewards.append(base + noise)
    
    analyzer = MetricsAnalyzer()
    results = analyzer.analyze_algorithm(test_rewards)
    
    print("Test Analysis Results:")
    print(f"Final Performance: {results['final_performance']['mean']:.2f}")
    print(f"Convergence Episode: {results['convergence']['convergence_episode']}")
    print(f"Volatility: {results['stability']['volatility']:.3f}")
    print(f"Trend: {results['learning_curve']['trend']}")
