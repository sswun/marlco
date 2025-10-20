# MADDPG (Multi-Agent Deep Deterministic Policy Gradient)

## 算法简介

MADDPG（多智能体深度确定性策略梯度）是一种专为多智能体环境设计的强化学习算法。它通过**"集中式训练，分布式执行"（CTDE）**的框架，有效解决了多智能体系统中的非平稳性问题。

### 核心思想

1. **集中式训练**: 训练时，每个智能体的Critic网络可以访问所有智能体的观测和动作，从而获得稳定的学习环境
2. **分布式执行**: 执行时，每个智能体的Actor网络仅依赖自己的局部观测进行决策
3. **Actor-Critic架构**: 
   - Actor网络输出动作概率分布
   - Critic网络评估状态-动作对的价值

### 与QMIX的区别

| 特性 | MADDPG | QMIX |
|------|--------|------|
| **网络结构** | Actor-Critic (每个智能体独立) | Q-learning + Mixing Network |
| **动作选择** | 策略梯度（连续/离散动作） | ε-greedy Q值选择 |
| **集中式信息** | Critic使用全局观测+动作 | Mixing Network组合个体Q值 |
| **适用场景** | 协作/竞争/混合 | 主要适用于协作任务 |
| **探索策略** | 高斯噪声/OU噪声 | ε-greedy |

## 项目结构

```
maddpg/
├── src/
│   ├── __init__.py           # 包初始化
│   ├── models.py             # Actor和Critic网络模型
│   ├── algos.py              # MADDPG算法核心实现
│   ├── buffer.py             # 经验回放缓冲区
│   ├── envs.py               # 环境包装器
│   ├── trainer.py            # 训练器
│   ├── utils.py              # 工具函数
│   └── pettingzoo_adapter.py # PettingZoo环境适配器
├── checkpoints/              # 模型检查点保存目录
├── plots/                    # 训练图表保存目录
├── config.yaml               # 默认配置文件
├── config_*.yaml             # 各环境专用配置
├── main.py                   # 主训练脚本
└── README.md                 # 本文件
```

## 快速开始

### 1. 基本训练

使用默认配置训练CM环境:

```bash
cd maddpg
python main.py
```

### 2. 指定环境和难度

```bash
# DEM环境，难度normal
python main.py --env DEM --difficulty normal

# HRG环境，ultra_fast版本
python main.py --env HRG --difficulty ultra_fast

# MSFS环境，难度hard
python main.py --env MSFS --difficulty hard
```

### 3. 使用配置文件

```bash
# 使用CM hard配置
python main.py --config config_CM_hard.yaml

# 使用DEM normal配置
python main.py --config config_DEM_normal.yaml
```

### 4. 生成训练图表

```bash
# 训练并生成图表
python main.py --plots

# 训练并显示图表
python main.py --show-plots

# 指定图表保存目录
python main.py --plots --plot-dir my_plots
```

### 5. 自定义训练参数

```bash
# 训练10000个episodes
python main.py --episodes 10000

# 使用特定随机种子
python main.py --seed 123

# 组合使用
python main.py --env CM --difficulty hard --episodes 8000 --plots
```

## 配置说明

### 环境配置 (`env`)

```yaml
env:
  name: "CM"                    # 环境名称: DEM, HRG, MSFS, CM, SMAC
  difficulty: "hard"           # 难度级别
  global_state_type: "concat"  # 全局状态类型
```

### 算法配置 (`algorithm`)

```yaml
algorithm:
  gamma: 0.99                  # 折扣因子
  actor_lr: 0.001             # Actor学习率
  critic_lr: 0.001            # Critic学习率
  tau: 0.005                   # 目标网络软更新系数
  max_grad_norm: 10.0         # 梯度裁剪
```

### 模型配置 (`model`)

```yaml
model:
  actor_hidden_dim: 256        # Actor网络隐藏层维度
  critic_hidden_dim: 256       # Critic网络隐藏层维度
```

### 训练配置 (`training`)

```yaml
training:
  total_episodes: 5000        # 总训练episodes
  batch_size: 64              # 批处理大小
  buffer_size: 100           # 经验回放缓冲区大小（×1000）
  warmup_episodes: 50        # 预热episodes
  eval_interval: 100          # 评估间隔
  save_interval: 500          # 保存间隔
```

### 探索配置 (`exploration`)

```yaml
exploration:
  noise_type: "gaussian"      # 噪声类型: gaussian, ou
  noise_scale: 0.1           # 噪声标准差
  noise_decay: 0.995         # 噪声衰减率
  noise_min: 0.01            # 最小噪声
```

## 支持的环境

本实现支持以下多智能体环境:

### 1. 自定义环境
- **CM** (Cooperative Movement): 合作移动任务
  - 难度: easy, normal, hard
  
- **DEM** (Disaster Emergency Management): 灾害应急管理
  - 难度: easy, normal, hard
  
- **HRG** (Hospital Resource Grid): 医院资源调度
  - 难度: easy, normal, hard, ultra_fast
  
- **MSFS** (Multi-Sensor Fusion System): 多传感器融合
  - 难度: easy, normal, hard

### 2. SMAC (StarCraft Multi-Agent Challenge)
- 地图: 8m, 3s5z, MMM, 等
- 预设配置: easy, normal, hard, debug

### 3. PettingZoo环境
- **multiwalker**: 多步行者协作
- **simple_spread**: 简单扩散
- **simple_crypto**: 简单加密通信

## 训练输出

训练过程中会生成以下内容:

### 1. 检查点文件
保存在 `checkpoints/` 目录:
- `maddpg_episode_*.pt`: 模型检查点
- `*_training_data_*.json`: 训练数据

### 2. 训练图表
保存在 `plots/` 目录（需使用`--plots`参数）:
- Episode奖励曲线
- Episode长度统计
- 训练损失曲线
- 噪声衰减曲线
- 奖励分布直方图
- 性能趋势分析

### 3. 控制台日志
```
🌍 Environment: CM
   Agents: 3
   Obs dims: [12, 12, 12]
   Action dims: [5, 5, 5]
   Device: cuda

🚀 开始MADDPG训练...
Episode      0 | Avg Reward: -45.23 | Avg Length: 78.0 | Noise: 0.100 | Buffer:    50
Episode    100 | Avg Reward: -12.34 | Avg Length: 95.5 | Noise: 0.060 | Buffer:  5000
🎯 Evaluation at episode 100: -8.45
...
```

## 算法特点

### 优势
- ✅ 有效解决多智能体环境的非平稳性问题
- ✅ 支持分布式执行，通信成本低
- ✅ 适用于协作、竞争和混合型任务
- ✅ 可处理连续和离散动作空间

### 局限
- ❌ 智能体数量过多时存在可扩展性问题
- ❌ 训练时需要访问所有智能体的信息
- ❌ 样本效率相对较低

## 与QMIX的对比实验

为了算法对比，MADDPG与QMIX使用相同的:
- 环境配置
- 训练episodes数
- 评估间隔
- 数据保存格式
- 图表生成方式

可以直接对比两个算法在相同环境下的性能差异。

## 参考文献

1. Lowe, R., Wu, Y., Tamar, A., Harb, J., Abbeel, P., & Mordatch, I. (2017). Multi-agent actor-critic for mixed cooperative-competitive environments. In Advances in Neural Information Processing Systems (pp. 6379-6390).

2. Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez, T., Tassa, Y., ... & Wierstra, D. (2015). Continuous control with deep reinforcement learning. arXiv preprint arXiv:1509.02971.

## 许可证

本项目遵循与主MARL项目相同的许可证。
