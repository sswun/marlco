# VDN: Value Decomposition Networks

这是VDN (Value Decomposition Networks)的完整实现，基于教程`tutorials/basicmethods/VDN.md`，参考了QMIX文件夹中的实现结构。

## 算法概述

VDN (Value Decomposition Networks) 是一种用于协作型多智能体强化学习的值分解方法，由Sunehag等人在2017年提出。其核心思想是将复杂的联合动作价值函数分解为每个智能体局部价值函数的和：

$$Q_{tot}(\mathbf{\tau}, \mathbf{u}) = \sum_{i=1}^{n} Q_i(\tau_i, u_i)$$

### 关键特性

1. **值分解**: 使用简单求和进行值分解，满足IGM (Individual-Global-Max) 原则
2. **CTDE架构**: 中心化训练，去中心化执行
3. **DRQN网络**: 使用GRU处理部分可观测性
4. **经验回放**: 支持异构观测和动态状态维度

## 文件结构

```
vdn/
├── src/                    # 源代码
│   ├── __init__.py
│   ├── models.py          # VDN网络模型 (AgentNetwork, VDNNetworks)
│   ├── algos.py           # VDN算法核心实现
│   ├── buffer.py          # 经验回放缓冲区
│   ├── trainer.py         # 训练器
│   ├── utils.py           # 工具函数
│   └── envs.py            # 环境包装器
├── config.yaml            # 配置文件
├── main.py                 # 主训练脚本
├── test.py                 # 功能测试脚本
├── run_test.py            # 简单测试脚本
└── README.md              # 本文件
```

## 核心实现

### 1. 网络架构 (`src/models.py`)

- **AgentNetwork**: DRQN网络，包含GRU层处理序列数据
- **VDNNetworks**: VDN网络集合，管理所有智能体网络和目标网络
- **值分解**: 使用简单求和 `Q_total = Σ Q_i`

### 2. 算法核心 (`src/algos.py`)

- **VDN类**: 实现完整的VDN算法
- **动作选择**: epsilon-greedy策略
- **损失计算**: TD误差损失函数
- **目标网络**: 软更新机制

### 3. 经验回放 (`src/buffer.py`)

- **ReplayBuffer**: 支持异构观测维度
- **动态填充**: 自动处理不同维度的状态和观测
- **统计功能**: 记录训练统计信息

### 4. 环境支持 (`src/envs.py`)

支持所有现有的CTDE环境：
- **CM**: Collaborative Moving
- **DEM**: Defense, Escort, Movement
- **HRG**: Heterogeneous Resource Gathering
- **MSFS**: Multi-agent Smart Factory Scheduling
- **SMAC**: StarCraft Multi-Agent Challenge
- **PettingZoo**: multiwalker, simple_spread, simple_crypto

## 快速开始

### 1. 环境测试

```bash
cd vdn
python run_test.py
```

### 2. 完整功能测试

```bash
python test.py
```

### 3. 训练VDN

```bash
# 使用默认配置
python main.py

# 指定环境和难度
python main.py --env CM --difficulty hard --episodes 5000

# 生成图表
python main.py --plots --show-plots

# 使用GPU
python main.py --device cuda
```

## 配置说明

### 环境配置
```yaml
env:
  name: "CM"                    # 环境名称
  difficulty: "hard"           # 难度级别
  global_state_type: "concat"  # 全局状态类型
```

### 算法配置
```yaml
algorithm:
  gamma: 0.99                 # 折扣因子
  learning_rate: 0.001        # 学习率
  tau: 0.005                  # 目标网络软更新系数
  target_update_interval: 50  # 目标网络更新间隔
  max_grad_norm: 10.0         # 梯度裁剪
```

### 模型配置
```yaml
model:
  hidden_dim: 256             # 个体网络隐藏层维度
```

### 训练配置
```yaml
training:
  total_episodes: 5000        # 总训练episodes
  batch_size: 64              # 批处理大小
  buffer_size: 100            # 经验回放缓冲区大小
  warmup_episodes: 50         # 预热episodes
  eval_interval: 100          # 评估间隔
  save_interval: 500          # 保存间隔
```

### 探索配置
```yaml
exploration:
  epsilon_start: 0.3          # 初始探索率
  epsilon_end: 0.1            # 最终探索率
  epsilon_decay: 0.995        # 探索率衰减
```

## 与QMIX的对比

| 特性 | VDN | QMIX |
|------|-----|------|
| 值分解方式 | 简单求和 | 单调混合网络 |
| 表达能力 | 线性 | 非线性 |
| 复杂度 | 低 | 中等 |
| 适用场景 | 简单协作任务 | 复杂协作任务 |
| 实现难度 | 简单 | 中等 |

## 使用示例

### Python API

```python
import sys
import os
sys.path.append('path/to/vdn')

from src.utils import load_config, set_seed
from src.envs import create_env_wrapper
from src.algos import VDN
from src.trainer import VDNTrainer

# 加载配置
config = load_config('config.yaml')

# 设置随机种子
set_seed(42)

# 创建训练器
trainer = VDNTrainer(config)

# 开始训练
episode_rewards = trainer.train()

# 评估
eval_reward = trainer.evaluate(num_episodes=20)
print(f"评估奖励: {eval_reward:.2f}")

# 关闭
trainer.close()
```

### 命令行参数

```bash
python main.py --help
```

主要参数：
- `--config`: 配置文件路径
- `--env`: 环境名称
- `--difficulty`: 难度级别
- `--episodes`: 训练episodes数
- `--plots`: 生成训练图表
- `--device`: 计算设备 (cpu/cuda/auto)

## 输出文件

### 模型文件
- `checkpoints/{experiment}_episode_{N}.pth`: 模型检查点
- `checkpoints/{experiment}_state_{N}.json`: 训练状态

### 训练数据
- `checkpoints/{experiment}_training_data_{timestamp}.json`: 完整训练数据

### 图表文件
- `plots/`: 训练图表目录
  - 奖励曲线
  - 损失曲线
  - 评估结果
  - 探索率变化

## 日志文件

- `vdn_training.log`: 详细训练日志

## 依赖项

核心依赖：
- `torch`
- `numpy`
- `matplotlib`
- `yaml`
- `seaborn`

环境依赖（根据使用的环境）：
- `gymnasium`
- `pygame`
- `pettingzoo`

## 性能特点

### 优势
1. **简洁高效**: 算法简单，计算复杂度低
2. **稳定训练**: 线性分解避免复杂的优化问题
3. **理论保证**: 满足IGM原则，保证去中心化最优性
4. **易于实现**: 代码结构清晰，易于理解和修改

### 局限性
1. **表达能力有限**: 线性假设无法表示复杂的协同效应
2. **任务适应性**: 在需要紧密协同的任务中表现可能受限
3. **全局信息利用**: 训练时也无法使用全局状态信息

## 扩展和定制

### 添加新环境
1. 在`src/envs.py`中添加环境创建逻辑
2. 确保环境提供CTDE接口
3. 更新配置文件

### 修改网络架构
1. 在`src/models.py`中修改`AgentNetwork`
2. 调整`VDNNetworks`中的网络创建逻辑
3. 更新配置文件中的模型参数

### 自定义训练流程
1. 继承`VDNTrainer`类
2. 重写相关方法
3. 在`main.py`中使用自定义训练器

## 故障排除

### 常见问题

1. **导入错误**: 确保路径设置正确
2. **CUDA错误**: 检查PyTorch和CUDA版本兼容性
3. **环境错误**: 确保环境依赖已正确安装
4. **内存不足**: 减小批次大小或缓冲区大小

### 调试技巧

1. 使用`run_test.py`进行快速功能验证
2. 查看`vdn_training.log`了解详细错误信息
3. 使用小规模配置进行快速测试

## 引用

如果您使用了此VDN实现，请引用：

```bibtex
@inproceedings{sunehag2017valuedecomposition,
  title={Value-decomposition networks for cooperative multi-agent learning},
  author={Sunehag, Peter and Lever, Guy and Gruslys, Audrius and
          Lazaric, Alessandro and Graepel, Thore},
  booktitle={International Conference on Machine Learning},
  pages={2085--2094},
  year={2017},
  organization={PMLR}
}
```

## 联系方式

如有问题或建议，请通过以下方式联系：
- 项目Issues
- 代码审查和改进建议