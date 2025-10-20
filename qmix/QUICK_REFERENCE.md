# 快速参考 - 训练数据与绘图

## 🚀 常用命令

### 训练相关

```bash
# 普通训练（自动保存数据）
python main.py

# 训练+立即绘图
python main.py --plots

# 训练+显示图表
python main.py --plots --show-plots

# 指定环境和难度
python main.py --env DEM --difficulty hard --episodes 2000

# SMAC 环境训练
python main.py --env SMAC --difficulty 8m --episodes 5000

# SMAC 使用预定义配置
python main.py --env SMAC --difficulty normal
```

### 绘图相关

```bash
# 查看所有保存的训练数据
python plot_from_data.py --list

# 使用最新数据绘图
python plot_from_data.py --latest

# 使用指定文件绘图
python plot_from_data.py --file checkpoints/xxx.json

# 查看数据摘要
python plot_from_data.py --file checkpoints/xxx.json --summary

# 绘图并显示
python plot_from_data.py --latest --show
```

## 📂 文件位置

| 文件类型 | 位置 | 格式 |
|---------|------|------|
| 训练数据 | `checkpoints/` | `{env}_{difficulty}_training_data_{timestamp}.json` |
| 图表 | `plots/` | `{env}_{metric}_{timestamp}.png` |
| 模型 | `checkpoints/` | `qmix_episode_{n}.pt` |

## 🎨 生成的图表

| # | 图表名称 | 说明 |
|---|---------|------|
| 1 | Episode Rewards | 训练奖励曲线 |
| 2 | Episode Lengths | Episode长度 |
| 3 | Training Loss | 训练损失 |
| 4 | Epsilon Decay | 探索率衰减 |
| 5 | Reward Histogram | 奖励分布 |
| 6 | Reward Boxplot | 奖励箱线图 |
| 7 | Learning Curves | 学习曲线 |
| 8 | Performance Trend | 性能趋势 |
| 9 | Recent Distribution | 近期分布 |
| 10 | Training Summary | 训练摘要 |

## 💡 快速技巧

### 找到最新的训练数据
```bash
ls -lt checkpoints/*_training_data_*.json | head -1
```

### 批量绘图
```bash
for file in checkpoints/*_training_data_*.json; do
    python plot_from_data.py --file "$file" --plot-dir "plots/$(basename $file .json)"
done
```

### 查看数据文件内容
```bash
cat checkpoints/xxx.json | python -m json.tool | less
```

## 🔧 Python API

### 在代码中使用

```python
from src.utils import (
    load_training_data,
    plot_from_file,
    list_training_data_files,
    print_training_data_summary
)

# 加载数据
data = load_training_data('checkpoints/xxx.json')

# 绘制图表
plot_files = plot_from_file('checkpoints/xxx.json', save_dir='plots')

# 列出文件
files = list_training_data_files('checkpoints')

# 打印摘要
print_training_data_summary('checkpoints/xxx.json')
```

## ⚡ 故障排除

| 问题 | 解决方案 |
|-----|---------|
| 找不到数据文件 | `python plot_from_data.py --list` |
| 图表生成失败 | `python plot_from_data.py --file xxx.json --summary` |
| 没有plots目录 | 会自动创建 |
| JSON格式错误 | 检查文件是否完整，可能训练中断 |

## 📚 相关文档

- **详细指南**: [`PLOTTING_GUIDE.md`](PLOTTING_GUIDE.md)
- **功能总结**: [`DATA_SAVING_SUMMARY.md`](DATA_SAVING_SUMMARY.md)
- **主配置**: [`config.yaml`](config.yaml)
