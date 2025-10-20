# COMA 算法实现

这是 COMA (Counterfactual Multi-Agent Policy Gradients) 算法的完整实现，基于论文 "Counterfactual Multi-Agent Policy Gradients" (Foerster et al., 2018)。

## 特性

- **反事实优势函数**: 实现了COMA的核心创新，通过反事实基线解决信用分配问题
- **中心化训练，去中心化执行(CTDE)**: 训练时使用全局信息，执行时仅使用局部观测
- **高效的Critic网络**: 一次前向传播计算所有可能动作的Q值，避免重复计算
- **与QMIX完全兼容**: 使用相同的配置格式、环境接口和训练流程
- **完整的训练流程**: 包含评估、保存、可视化等完整功能

## 文件结构

```
coma/
├── src/
│   ├── __init__.py          # 包初始化
│   ├── algos.py            # COMA算法核心实现
│   ├── models.py           # Actor-Critic网络架构
│   ├── buffer.py           # 经验回放缓冲区
│   ├── trainer.py          # 训练器主循环
│   ├── utils.py            # 工具函数
│   └── envs.py             # 环境包装器
├── config.yaml             # 训练配置文件
├── main.py                 # 主训练脚本
├── test.py                 # 完整测试套件
├── debug_test.py           # 简单调试测试
└── README.md               # 本文件
```

## 核心算法特性

### 1. 反事实优势函数

COMA的核心创新是反事实优势函数，用于精确评估每个智能体的贡献：

```
A^a(s, τ, u) = Q(s, τ, u) - Σ_{u'^a} π^a(u'^a | τ^a) Q(s, τ, (u^{-a}, u'^a))
```

- 第一项：当前联合动作的实际价值
- 第二项：反事实基线（如果只有智能体a随机行动的期望价值）
- 差值：智能体a执行当前动作的边际贡献

### 2. 高效的Critic网络设计

- **输入**: 全局状态 + 所有智能体观测 + 其他智能体动作
- **输出**: 指定智能体对所有可能动作的Q值向量
- **优势**: 一次前向传播获得所有动作的Q值，计算复杂度从O(|U|^n)降低到O(|U|)

### 3. Actor-Critic架构

- **Actor网络**: 每个智能体的策略网络，输出动作概率分布
- **Critic网络**: 中心化评论家，评估联合动作价值
- **参数共享**: 所有智能体共享网络参数，提高学习效率

## 使用方法

### 1. 基本训练

```bash
# 使用默认配置训练
python main.py

# 使用特定配置文件（完全兼容QMIX配置）
python main.py --config config_CM_hard.yaml

# 指定环境和参数
python main.py --env CM --difficulty easy --episodes 1000

# 调整学习率和其他参数
python main.py --env HRG --lr 0.0005 --gamma 0.95
```

### 2. QMIX兼容的命令行参数

```bash
# 完全兼容QMIX的参数格式
python main.py --config config_CM_hard.yaml --plots --plot-dir CMhardplots

# 主要参数说明
--config             配置文件路径（默认：config.yaml）
--env                环境名称 (覆盖配置文件) [CM, DEM, HRG, MSFS, SMAC, multiwalker, simple_spread, simple_crypto]
--difficulty         难度级别/地图名称 (覆盖配置文件)
--episodes           训练episodes数 (覆盖配置文件)
--seed               随机种子 (默认：42)
--plots              生成并保存训练图表
--show-plots         训练后显示图表
--plot-dir           图表保存目录 (默认：plots)
```

### 3. 支持的配置文件

COMA完全兼容所有QMIX配置文件：

```bash
# 标准环境配置
python main.py --config config_CM_hard.yaml --plots
python main.py --config config_DEM_hard.yaml --plots
python main.py --config config_HRG_ultrafast.yaml --plots
python main.py --config config_MSFS_hard.yaml --plots

# PettingZoo环境配置
python main.py --config config_simple_spread.yaml --plots
python main.py --config config_simple_crypto.yaml --plots
```

### 4. 配置文件

主要配置项在 `config.yaml` 中：

```yaml
env:
  name: "CM"                    # 环境名称
  difficulty: "hard"           # 难度设置
  global_state_type: "concat"  # 全局状态类型

algorithm:
  gamma: 0.99                  # 折扣因子
  learning_rate: 0.001        # 学习率
  lambda: 0.8                  # TD(λ)参数

model:
  hidden_dim: 256             # 网络隐藏层维度

training:
  total_episodes: 5000        # 总训练episodes
  batch_size: 64              # 批处理大小
  buffer_size: 100            # 经验回放缓冲区大小

exploration:
  epsilon_start: 0.3          # 初始探索率
  epsilon_end: 0.1            # 最终探索率
  epsilon_decay: 0.995        # 探索率衰减
```

### 4. 测试

```bash
# 运行完整测试套件
python test.py

# 运行简单调试测试
python debug_test.py
```

## 算法对比

| 特性 | COMA | QMIX | MADDPG |
|------|------|------|--------|
| **核心思想** | 反事实优势函数 | 值函数分解 | 独立Actor-Critic |
| **信用分配** | 反事实基线 | 单调值函数分解 | 个体价值函数 |
| **网络架构** | Actor-Critic | Q值网络 | Actor-Critic |
| **训练范式** | CTDE | CTDE | CTDE |
| **优势** | 精确信用分配 | 最优性保证 | 通用性强 |
| **适用场景** | 合作任务 | 合作任务 | 混合动机 |

## 实现细节

### 环境兼容性

支持所有MARL环境：
- **CM**: Collaborative Moving - 合作推箱子
- **DEM**: Defense, Escort, Movement - VIP保护
- **HRG**: Heterogeneous Resource Gathering - 异构资源收集
- **MSFS**: Multi-agent Smart Factory Scheduling - 智能工厂调度
- **SMAC**: StarCraft Multi-Agent Challenge - 星际争霸微操

### 训练监控

- 实时显示训练进度和性能指标
- 定期评估和模型保存
- 自动生成训练图表（需要matplotlib）
- 保存训练数据用于后续分析

### 性能优化

- GPU加速训练
- 高效的经验回放缓冲区
- 批量处理和向量化计算
- 参数共享减少内存占用

## 实验结果

COMA在多个合作任务中表现优异：

1. **精确信用分配**: 反事实基线使每个智能体能准确评估自己的贡献
2. **快速收敛**: 优势函数提供清晰的学习信号
3. **稳定训练**: 梯度趋向于零，避免策略振荡
4. **可扩展性**: 网络复杂度随智能体数量线性增长

## 与QMIX的对比

本实现确保了与QMIX的完全兼容：

- **相同的环境接口**: 支持所有相同的环境
- **一致的配置格式**: 使用相同的YAML配置结构
- **兼容的训练流程**: 相同的训练、评估、保存流程
- **可比的性能指标**: 便于算法效果对比

## 引用

如果您使用了本实现，请引用原始论文：

```bibtex
@inproceedings{foerster2018counterfactual,
  title={Counterfactual Multi-Agent Policy Gradients},
  author={Foerster, Jakob and Assael, Yannis M and de Freitas, Nando and Whiteson, Shimon},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  pages={2974--2982},
  year={2018}
}
```

## 许可证

本实现遵循MIT许可证。