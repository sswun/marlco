# 训练数据保存与绘图功能总结

## ✅ 已完成的功能

### 1. 训练数据自动保存 (`trainer.py`)

新增方法 `save_training_data()`：
- 自动保存所有训练指标到JSON文件
- 包含完整的配置信息
- 包含环境信息
- 文件命名: `{环境}_{难度}_training_data_{时间戳}.json`
- 保存位置: `checkpoints/` 目录

**保存的数据**:
```python
{
    'config': 完整的训练配置,
    'metrics': {
        'episode_rewards': 每个episode的奖励,
        'episode_lengths': 每个episode的长度,
        'losses': 训练损失,
        'eval_episodes': 评估的episode编号,
        'eval_rewards': 评估奖励,
        'epsilon_history': 探索率历史
    },
    'environment': 环境详细信息,
    'timestamp': 时间戳,
    'total_episodes': 总episode数
}
```

### 2. 从文件加载数据 (`utils.py`)

新增三个核心函数：

#### `load_training_data(filepath)`
- 从JSON文件加载训练数据
- 自动验证文件格式
- 返回完整的数据字典

#### `plot_from_file(filepath, save_dir, show_plots)`
- 从数据文件直接生成所有图表
- 自动提取环境名称和配置
- 返回生成的图表文件列表

#### `list_training_data_files(directory)`
- 列出指定目录中的所有训练数据文件
- 按时间倒序排列（最新的在前）
- 返回文件路径列表

#### `print_training_data_summary(filepath)`
- 打印数据文件的详细摘要
- 显示性能指标统计
- 显示评估结果

### 3. 独立绘图脚本 (`plot_from_data.py`)

全功能的命令行工具：

```bash
# 列出所有数据文件
python plot_from_data.py --list

# 使用最新数据绘图
python plot_from_data.py --latest

# 使用指定文件绘图
python plot_from_data.py --file checkpoints/xxx.json

# 显示数据摘要
python plot_from_data.py --file checkpoints/xxx.json --summary

# 显示图表（不仅保存）
python plot_from_data.py --latest --show

# 自定义保存目录
python plot_from_data.py --latest --plot-dir my_plots
```

### 4. 主训练脚本集成 (`main.py`)

训练完成后自动保存数据：
- 训练成功完成 → 保存数据
- 用户中断（Ctrl+C） → 仍保存数据
- 训练出错 → 尝试保存已有数据

### 5. 详细使用文档 (`PLOTTING_GUIDE.md`)

包含：
- 功能概述
- 完整的使用示例
- 常见用例
- 故障排除
- 数据格式说明

## 📁 文件结构

```
marl/
├── checkpoints/                    # 模型和数据保存目录
│   ├── qmix_episode_xxx.pt        # 模型检查点
│   └── *_training_data_*.json     # 训练数据
├── plots/                          # 图表保存目录
│   ├── *_episode_rewards_*.png
│   ├── *_training_loss_*.png
│   └── ...
├── src/
│   ├── trainer.py                 # 新增: save_training_data()
│   └── utils.py                   # 新增: 数据加载和绘图函数
├── main.py                        # 修改: 集成数据保存
├── plot_from_data.py              # 新增: 独立绘图脚本
└── PLOTTING_GUIDE.md              # 新增: 使用指南
```

## 🎯 使用工作流

### 标准训练流程

```bash
# 1. 训练（自动保存数据）
python main.py --env DEM --difficulty normal --episodes 1000

# 2. 训练会自动保存数据到 checkpoints/
# 输出: DEM_normal_training_data_20231015_153045.json

# 3. 如需绘图，使用 plot_from_data.py
python plot_from_data.py --latest
```

### 带实时绘图的训练

```bash
# 训练并立即生成图表
python main.py --plots --plot-dir plots
```

### 后续分析

```bash
# 查看所有历史训练
python plot_from_data.py --list

# 为特定训练生成图表
python plot_from_data.py --file checkpoints/DEM_normal_training_data_xxx.json

# 查看训练摘要
python plot_from_data.py --file checkpoints/DEM_normal_training_data_xxx.json --summary
```

## 🎨 生成的图表

每次绘图生成约10-12个图表：

### 训练进度类
1. Episode Rewards（奖励曲线）
2. Episode Lengths（长度曲线）
3. Training Loss（损失曲线）
4. Epsilon Decay（探索率衰减）

### 统计分析类
5. Reward Histogram（奖励分布直方图）
6. Reward Boxplot（奖励箱线图）

### 性能分析类
7. Learning Curves（多窗口学习曲线）
8. Performance Trend（性能趋势）
9. Recent Distribution（近期表现分布）

### 摘要类
10. Training Summary（训练摘要统计）
11. Plot Index（图表索引文件）

## 💡 优势

1. **数据持久化**: 训练数据永久保存，可随时重新分析
2. **灵活绘图**: 可以多次从同一数据生成不同的图表
3. **版本控制**: JSON格式便于版本管理和对比
4. **独立性**: 绘图与训练解耦，不影响训练性能
5. **易于分享**: JSON文件体积小，便于分享和复现
6. **高质量**: 300 DPI图表适合论文发表
7. **自动化**: 训练完成自动保存，无需手动操作

## 🔧 技术细节

### 数据格式
- **格式**: JSON
- **编码**: UTF-8
- **大小**: 约50-500 KB（取决于训练长度）
- **可读性**: 人类可读，便于调试

### 图表格式
- **格式**: PNG
- **分辨率**: 300 DPI
- **尺寸**: 12×8 英寸
- **样式**: Seaborn v0.8

### 兼容性
- Python 3.8+
- 所有主流操作系统
- 可在无GUI环境运行（--show除外）

## 📝 示例输出

### 训练完成时
```
✅ 训练完成! 总时间: 1234.5s

📊 训练结果统计:
   最终平均奖励: 42.15
   总episodes: 1000
   最终评估分数: 48.23

💾 保存训练数据...
💾 训练数据已保存: checkpoints/DEM_normal_training_data_20231015_153045.json
✅ 数据文件已保存
```

### 从文件绘图时
```
📊 开始从数据文件绘制图表...
   环境: DEM_normal
   总 episodes: 1000
   保存目录: plots

✅ 绘图完成! 生成了 11 个图表
📁 图表保存在: plots
```

## 🎓 最佳实践

1. **命名规范**: 数据文件自动包含环境、难度和时间戳
2. **定期清理**: 定期清理旧的数据文件和图表
3. **备份重要数据**: 将关键实验的数据文件备份
4. **版本控制**: 将训练数据加入Git（JSON文件小）
5. **文档化**: 在README中记录重要训练的数据文件名

## 🚀 未来可能的扩展

- [ ] 支持多个训练的对比图表
- [ ] 交互式图表（使用Plotly）
- [ ] 导出为PDF报告
- [ ] 自动生成训练报告
- [ ] 支持TensorBoard格式
- [ ] 数据压缩选项

## ✨ 总结

现在你可以：
1. ✅ 训练时自动保存所有数据
2. ✅ 随时从数据文件重新生成图表
3. ✅ 查看历史训练的摘要和统计
4. ✅ 对比不同训练的结果
5. ✅ 生成高质量的论文图表
6. ✅ 完全解耦训练和可视化过程

所有功能都已集成到现有代码中，无需额外配置即可使用！
