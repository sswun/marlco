"""
MARL Training Results Analysis System

A comprehensive toolkit for analyzing multi-agent reinforcement learning training results.

Modules:
    - data_loader: Load and parse training data from checkpoints
    - metrics_analyzer: Compute performance metrics
    - plot_generator: Generate publication-quality plots
    - analyze_results: Main analysis pipeline

Example:
    >>> from analyze_results import ResultsAnalyzer
    >>> analyzer = ResultsAnalyzer()
    >>> analyzer.run_complete_analysis()
"""

__version__ = "1.0.0"
__author__ = "MARL Analysis Team"

from .data_loader import TrainingDataLoader, compute_statistics
from .metrics_analyzer import MetricsAnalyzer
from .plot_generator import PlotGenerator
from .analyze_results import ResultsAnalyzer

__all__ = [
    'TrainingDataLoader',
    'compute_statistics',
    'MetricsAnalyzer',
    'PlotGenerator',
    'ResultsAnalyzer'
]
