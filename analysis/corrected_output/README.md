# MARLCO Benchmark Results Visualization

This directory contains comprehensive benchmark results and visualizations for all algorithms across different environments.

## 📊 Available Visualizations

### Learning Curves (by Environment)

Episode reward curves with running mean and confidence intervals:

- `corrected_learning_curves_MSFS_hard.png` - Manufacturing (Hard)
- `corrected_learning_curves_MSFS_normal.png` - Manufacturing (Normal)
- `corrected_learning_curves_CM_hard.png` - Collaborative Moving (Hard)
- `corrected_learning_curves_HRG_ultrafast.png` - Resource Gathering (Ultra-fast)
- `corrected_learning_curves_DEM_hard.png` - Escort Mission (Hard)
- `corrected_learning_curves_DEM_normal.png` - Escort Mission (Normal)
- `corrected_learning_curves_simple_spread.png` - Simple Spread (PettingZoo)
- `corrected_learning_curves_multiwalker.png` - Multiwalker (PettingZoo)
- `corrected_learning_curves_simple_crypto.png` - Simple Crypto (PettingZoo)

### Performance Comparisons

- `corrected_normalized_performance_comparison.png` - Normalized performance across all environments
- `enhanced_performance_heatmap.png` - Performance heatmap with rankings
- `performance_distribution_analysis.png` - Box plots and statistical analysis

### Data Files

- `corrected_analysis_summary.txt` - Text summary of all results
- `corrected_detailed_metrics.json` - Detailed metrics in JSON format

## 🎯 Key Findings

### Overall Rankings

1. 🥇 **MAPPO** (0.778) - Best overall performance
2. 🥈 **IQL** (0.645) - Strong and stable
3. 🥉 **COMA** (0.510) - Good credit assignment

### Environment-Specific Insights

**MSFS (Convergence Test)**:
- ✅ All algorithms converge successfully
- 📈 QMIX achieves highest scores (86.27±2.60)

**CM & HRG (Challenging)**:
- ⚠️ Require longer training
- 📊 High variance in performance

**PettingZoo Environments**:
- 🎮 MAPPO excels in simple_spread
- 🤝 Consistent convergence in multiwalker

## 🔬 How to Reproduce

```bash
cd analysis
python corrected_analyze_results.py
```

This will regenerate all visualizations and analysis files.

## 📖 References

See the main [project README](../../README.md) for detailed algorithm descriptions and usage instructions.
