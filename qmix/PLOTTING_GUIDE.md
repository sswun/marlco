# 训练数据保存与绘图使用指南

## 概述

训练过程会自动保存训练数据到JSON文件，你可以随时从这些文件重新生成图表。

## 文件说明

### 训练数据文件
- **位置**: `checkpoints/` 目录
- **格式**: `{环境名}_{难度}_training_data_{时间戳}.json`
- **内容**: 
  - 训练配置
  - 所有训练指标（奖励、长度、损失等）
  - 环境信息
  - 评估结果

### 图表文件
- **位置**: `plots/` 目录（可自定义）
- **格式**: PNG，300 DPI 高分辨率
- **类型**:
  - Episode Rewards（episode奖励曲线）
  - Episode Lengths（episode长度）
  - Training Loss（训练损失）
  - Epsilon Decay（探索率衰减）
  - Reward Distribution（奖励分布）
  - Training Summary（训练摘要）
  - Performance Trends（性能趋势）

## 使用方法

### 1. 训练时自动保存数据

训练完成后会自动保存数据：

```bash
# 正常训练（会自动保存数据）
python main.py

# 训练并立即生成图表
python main.py --plots --plot-dir plots
```

### 2. 从保存的数据生成图表

#### 列出所有数据文件

```bash
python plot_from_data.py --list
```

输出示例：
```
📁 可用的训练数据文件:

   1. DEM_normal_training_data_20231015_153045.json
      大小: 125.3 KB
      路径: checkpoints/DEM_normal_training_data_20231015_153045.json

   2. CM_normal_training_data_20231015_120130.json
      大小: 98.7 KB
      路径: checkpoints/CM_normal_training_data_20231015_120130.json

总计: 2 个文件
```

#### 使用最新的数据文件绘图

```bash
python plot_from_data.py --latest
```

#### 使用指定的数据文件绘图

```bash
python plot_from_data.py --file checkpoints/DEM_normal_training_data_20231015_153045.json
```

#### 显示数据摘要

```bash
python plot_from_data.py --file checkpoints/DEM_normal_training_data_20231015_153045.json --summary
```

输出示例：
```
============================================================
训练数据摘要: DEM_normal_training_data_20231015_153045.json
============================================================

🎮 环境信息:
   名称: DEM
   难度: normal
   智能体数: 3

📊 训练统计:
   总 episodes: 1000
   时间戳: 20231015_153045

🎯 性能指标:
   平均奖励: 25.34
   最佳奖励: 78.50
   最终奖励: 45.20
   标准差: 18.76

   最近100 episodes:
      平均: 42.15
      最佳: 78.50

🎯 评估结果:
   评估次数: 10
   平均分数: 48.23
   最佳分数: 65.80
============================================================
```

#### 显示图表（不仅仅保存）

```bash
python plot_from_data.py --latest --show
```

#### 自定义保存目录

```bash
python plot_from_data.py --latest --plot-dir my_plots
```

## 数据文件格式

训练数据JSON文件包含以下内容：

```json
{
  "config": {
    "env": {...},
    "training": {...},
    "algorithm": {...}
  },
  "metrics": {
    "episode_rewards": [1.5, 2.3, ...],
    "episode_lengths": [50, 45, ...],
    "losses": [0.5, 0.4, ...],
    "eval_episodes": [100, 200, ...],
    "eval_rewards": [25.3, 30.5, ...],
    "epsilon_history": [1.0, 0.99, ...]
  },
  "environment": {
    "name": "DEM",
    "difficulty": "normal",
    "n_agents": 3,
    "obs_dims": [52, 52, 52],
    "act_dims": [10, 10, 10]
  },
  "timestamp": "20231015_153045",
  "total_episodes": 1000
}
```

## 生成的图表类型

### 1. Training Progress（训练进度）
- **Episode Rewards**: 显示训练过程中的奖励变化
- **Episode Lengths**: 显示每个episode的长度
- **Training Loss**: 显示训练损失的变化
- **Epsilon Decay**: 显示探索率的衰减过程

### 2. Statistical Analysis（统计分析）
- **Reward Histogram**: 奖励分布直方图
- **Reward Boxplot**: 奖励箱线图，显示分位数

### 3. Performance Analysis（性能分析）
- **Learning Curves**: 不同窗口大小的学习曲线
- **Performance Trend**: 性能趋势分析
- **Recent Distribution**: 最近episodes的性能分布

### 4. Summary（摘要）
- **Training Summary**: 训练配置和关键指标摘要

## 常见用例

### 对比不同训练的结果

```bash
# 生成训练1的图表
python plot_from_data.py --file checkpoints/DEM_normal_training_data_20231015_120000.json --plot-dir plots/run1

# 生成训练2的图表  
python plot_from_data.py --file checkpoints/DEM_normal_training_data_20231015_150000.json --plot-dir plots/run2

# 然后可以对比两个目录中的图表
```

### 中断训练后重新绘图

即使训练被中断，数据仍然会被保存（如果在最后保存）。你可以：

```bash
# 查看保存的数据
python plot_from_data.py --list

# 为中断的训练生成图表
python plot_from_data.py --latest --plot-dir plots/interrupted
```

### 生成论文/报告用的图表

```bash
# 生成高质量图表（300 DPI）
python plot_from_data.py --file checkpoints/best_run.json --plot-dir paper_figures
```

## 提示

1. **保存频率**: 训练数据在训练完成时自动保存
2. **文件大小**: 每个数据文件约几十到几百KB，取决于训练长度
3. **图表质量**: 所有图表都以300 DPI保存，适合论文发表
4. **多次绘图**: 可以多次从同一数据文件生成图表，不影响原始数据
5. **版本控制**: 建议将训练数据文件加入版本控制（如Git），方便追踪实验

## 故障排除

### 找不到数据文件
```bash
# 检查checkpoints目录
ls checkpoints/*_training_data_*.json

# 使用--list查看
python plot_from_data.py --list
```

### 图表生成失败
```bash
# 先查看数据摘要，检查数据是否完整
python plot_from_data.py --file your_data.json --summary
```

### 自定义数据目录
```bash
python plot_from_data.py --data-dir my_checkpoints --list
```
