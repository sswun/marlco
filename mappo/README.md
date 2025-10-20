# MAPPO (Multi-Agent Proximal Policy Optimization)

## 算法简介

MAPPO（多智能体近端策略优化）是PPO算法在多智能体环境中的扩展。它采用**"集中式训练，分布式执行"（CTDE）**框架，结合PPO的稳定性优势，实现高效的多智能体协作学习。

### 核心思想

1. **集中式Critic**: 训练时使用全局状态信息，评估策略价值
2. **分布式Actor**: 执行时每个智能体仅依赖局部观测
3. **PPO优化**: 使用重要性采样和裁剪机制，确保策略更新稳定
4. **GAE优势估计**: 使用广义优势估计(GAE)降低方差

### 与其他算法的区别

| 特性 | MAPPO | QMIX | MADDPG |
|------|-------|------|--------|
| **算法类型** | On-policy (PPO) | Value-Based | Off-policy (DDPG) |
| **训练稳定性** | 高 | 中 | 中 |
| **样本效率** | 低 | 中 | 高 |
| **Critic结构** | 集中式V函数 | Mixing Network | 集中式Q函数 |
| **探索策略** | 策略熵 | ε-greedy | 噪声注入 |
| **适用场景** | 协作任务 | 协作任务 | 协作/竞争/混合 |

## 项目结构

```
mappo/
├── src/
│   ├── __init__.py           # 包初始化
│   ├── models.py             # Actor和Critic网络模型
│   ├── algos.py              # MAPPO算法核心实现
│   ├── trainer.py            # 训练器
│   ├── envs.py               # 环境包装器
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

```bash
cd mappo
conda activate pettingzoo
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

## 配置说明

### 算法配置 (`algorithm`)

```yaml
algorithm:
  gamma: 0.99                  # 折扣因子
  actor_lr: 0.0003            # Actor学习率
  critic_lr: 0.001            # Critic学习率
  gae_lambda: 0.95            # GAE lambda参数
  clip_param: 0.2             # PPO裁剪参数
  value_loss_coef: 0.5        # 价值函数损失系数
  entropy_coef: 0.01          # 熵正则化系数
  max_grad_norm: 10.0         # 梯度裁剪
  ppo_epochs: 4               # PPO更新轮数
  num_mini_batch: 1           # mini-batch数量
```

### 模型配置 (`model`)

```yaml
model:
  actor_hidden_dim: 256                # Actor网络隐藏层维度
  critic_hidden_dim: 256               # Critic网络隐藏层维度
  use_feature_normalization: false     # 是否使用特征归一化
  use_orthogonal_init: true            # 是否使用正交初始化
```

### 训练配置 (`training`)

```yaml
training:
  total_episodes: 5000        # 总训练episodes
  n_rollout_threads: 1        # 并行环境数量
  episode_length: 200         # 每个episode最大长度
  buffer_size: 200            # 回放缓冲区大小
  warmup_episodes: 50         # 预热episodes
  eval_interval: 100          # 评估间隔
  save_interval: 500          # 保存间隔
```

## 核心算法

### PPO损失函数

**Actor损失**:
```
L_CLIP = -min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
其中:
  ratio = π_new(a|s) / π_old(a|s)
  A = 优势函数
  ε = 裁剪参数
```

**Critic损失**:
```
L_V = (V(s) - V_target)²
```

**总损失**:
```
L = L_CLIP + c1 * L_V - c2 * H(π)
其中:
  H(π) = 策略熵（促进探索）
  c1 = 价值函数损失系数
  c2 = 熵系数
```

### GAE优势估计

```
A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
其中:
  δ_t = r_t + γV(s_{t+1}) - V(s_t)
  λ = GAE lambda参数
```

## 支持的环境

- **CM, DEM, HRG, MSFS**: 自定义多智能体环境
- **SMAC**: 星际争霸多智能体挑战
- **PettingZoo**: multiwalker, simple_spread, simple_crypto

## 算法特点

### 优势
- ✅ 训练稳定性高（PPO裁剪机制）
- ✅ 不需要精细调参
- ✅ 适合协作任务
- ✅ 支持连续和离散动作空间
- ✅ 自然的探索机制（策略熵）

### 局限
- ❌ 样本效率相对较低（on-policy）
- ❌ 需要较多的环境交互
- ❌ 计算开销较大（多轮PPO更新）

## 与QMIX/MADDPG对比

MAPPO与QMIX、MADDPG使用相同的:
- 环境配置
- 训练episodes数
- 评估间隔
- 数据保存格式
- 图表生成方式

可直接对比三种算法的性能差异。

## 参考文献

1. Yu, C., Velu, A., Vinitsky, E., Wang, Y., Bayen, A., & Wu, Y. (2021). The surprising effectiveness of ppo in cooperative, multi-agent games. arXiv preprint arXiv:2103.01955.

2. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.

## 许可证

本项目遵循与主MARL项目相同的许可证。
