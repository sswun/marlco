# IQL (Independent Q-Learning) Implementation

## 概述

这是IQL（Independent Q-Learning）算法的完整实现，基于QMIX框架结构设计，用于多智能体强化学习研究。

### IQL算法特点

IQL是最基础的多智能体强化学习算法之一，其核心思想是：
- **去中心化训练，去中心化执行** (DTDE)
- **每个智能体独立学习**Q函数，将其他智能体视为环境的一部分
- **简单直观**，计算高效，适合理解多智能体学习的基本原理

## 文件结构

```
iql/
├── src/                           # 源代码目录
│   ├── __init__.py
│   ├── models.py                  # 神经网络模型
│   ├── algos.py                   # IQL算法核心实现
│   ├── buffer.py                  # 经验回放缓冲区
│   ├── trainer.py                 # 训练器
│   ├── envs.py                    # 环境包装器
│   ├── utils.py                   # 工具函数
│   └── pettingzoo_adapter.py      # PettingZoo环境适配器
├── config.yaml                    # 默认配置文件
├── config_DEM_normal.yaml         # DEM环境配置
├── main.py                        # 主训练脚本
├── test.py                        # 单元测试
├── run_test.py                    # 快速测试脚本
└── README.md                      # 本文档
```

## 核心组件

### 1. 算法实现 (`src/algos.py`)

IQL算法的核心特点是：
- **独立Q网络**：每个智能体有自己独立的Q网络和优化器
- **独立学习**：每个智能体基于自身观测和奖励独立更新Q值
- **无协作机制**：智能体之间没有信息交换或协作学习

```python
class IQL:
    """IQL算法实现 - Independent Q-Learning"""

    def compute_loss(self, batch):
        """为每个智能体独立计算损失"""
        total_loss = 0.0
        for i in range(self.networks.n_agents):
            # 每个智能体独立计算TD损失
            agent_loss = F.mse_loss(current_q, target_q)
            total_loss += agent_loss
        return total_loss / self.networks.n_agents
```

### 2. 网络模型 (`src/models.py`)

与QMIX的主要区别：
- **无混合网络**：IQL不需要值分解网络
- **独立目标网络**：每个智能体有自己的目标网络
- **支持异构观测**：可以处理不同智能体有不同观测维度的情况

```python
class IQLNetworks:
    """IQL网络集合 - 每个智能体有独立的Q网络和目标网络"""

    def __init__(self, env_info, config, device='cpu'):
        # 为每个智能体创建独立的Q网络
        self.agent_networks = nn.ModuleList([
            AgentNetwork(obs_dim, action_dim, hidden_dim)
            for _ in range(self.n_agents)
        ])
        # 为每个智能体创建独立的目标网络
        self.target_agent_networks = nn.ModuleList([...])
```

### 3. 经验缓冲区 (`src/buffer.py`)

虽然IQL不使用全局状态，但为了与QMIX框架保持兼容性，缓冲区保留了全局状态接口。

## 使用方法

### 1. 基本训练

```bash
# 使用默认配置训练
python main.py

# 指定环境和难度
python main.py --env CM --difficulty hard --episodes 5000

# 生成训练图表
python main.py --plots --show-plots

# 使用特定配置文件
python main.py --config config_DEM_normal.yaml
```

### 2. 快速测试

```bash
# 运行快速测试（20个episodes）
python run_test.py

# 运行完整的单元测试
python test.py
```

### 3. 配置参数

主要配置参数：

```yaml
algorithm:
  gamma: 0.99                    # 折扣因子
  learning_rate: 0.001          # 学习率
  target_update_interval: 50    # 目标网络更新间隔

model:
  hidden_dim: 256               # 网络隐藏层维度

training:
  total_episodes: 5000          # 总训练episodes
  batch_size: 64                # 批处理大小
  buffer_size: 100              # 经验缓冲区大小

exploration:
  epsilon_start: 0.3            # 初始探索率
  epsilon_end: 0.1             # 最终探索率
  epsilon_decay: 0.995          # 探索率衰减
```

## 支持的环境

IQL实现支持与QMIX相同的环境：

1. **CM** (Collaborative Moving) - 合作推箱子
2. **DEM** (Defense, Escort, Movement) - VIP保护
3. **HRG** (Heterogeneous Resource Gathering) - 异构资源收集
4. **MSFS** (Multi-agent Smart Factory Scheduling) - 智能工厂调度
5. **SMAC** (StarCraft Multi-Agent Challenge)
6. **PettingZoo环境** (multiwalker, simple_spread, simple_crypto)

## 算法特点对比

| 特性 | IQL | QMIX |
|------|-----|------|
| 训练方式 | 去中心化 | 中心化 |
| 执行方式 | 去中心化 | 去中心化 |
| 网络结构 | 独立Q网络 | Q网络 + 混合网络 |
| 信用分配 | 无 | 值分解 |
| 协作能力 | 有限 | 强 |
| 计算复杂度 | O(n) | O(n) |
| 适用场景 | 弱耦合任务 | 协作任务 |

## 优缺点分析

### 优势
- ✅ **简单直观**：易于理解和实现
- ✅ **计算高效**：每个智能体独立计算，可并行化
- ✅ **去中心化执行**：无需通信基础设施
- ✅ **理论基础**：为理解复杂MARL算法提供基础

### 局限性
- ❌ **非平稳性问题**：其他智能体策略变化导致环境不稳定
- ❌ **忽略智能体交互**：无法学习需要精确配合的策略
- ❌ **信用分配问题**：难以判断单个智能体对团队奖励的贡献
- ❌ **协作能力有限**：在复杂协作任务中性能受限

## 实验建议

### 1. 适用场景
- **弱耦合任务**：智能体间交互较少
- **同质智能体**：相同类型和目标的智能体
- **去中心化系统**：通信受限环境

### 2. 对比实验
可与以下算法进行对比：
- **QMIX**：验证值分解的优势
- **VDN**：简单的值分解方法
- **COMA**：基于actor-critic的方法

### 3. 性能评估
- 在简单任务中，IQL可能表现接近QMIX
- 在复杂协作任务中，IQL性能会显著下降
- 观察学习曲线的稳定性和收敛速度

## 技术细节

### 1. 与QMIX的兼容性
IQL实现保持了与QMIX框架的最大兼容性：
- 相同的环境接口
- 相同的配置格式
- 相同的训练流程
- 相同的可视化工具

### 2. 参数共享优化
支持参数共享的IQL变体：
```python
# 在config中启用参数共享
model:
  parameter_sharing: true  # 所有智能体共享网络参数
```

### 3. 异构环境支持
自动处理不同智能体的观测和动作空间：
- 检测异构观测维度
- 自动填充到统一维度
- 保持每个智能体的原始信息

## 故障排除

### 常见问题

1. **环境创建失败**
   - 检查环境名称和难度设置
   - 确保相应的环境模块已安装

2. **内存不足**
   - 减小`batch_size`和`buffer_size`
   - 使用较小的网络`hidden_dim`

3. **训练不稳定**
   - 调整学习率`learning_rate`
   - 增加预热episodes`warmup_episodes`
   - 调整探索率衰减`epsilon_decay`

4. **性能不佳**
   - 检查环境是否适合IQL（弱耦合）
   - 尝试更大的网络`hidden_dim`
   - 调整训练参数

## 扩展方向

基于当前IQL实现，可以进一步探索：

1. **参数共享IQL**：所有智能体共享网络参数
2. **IQL+RNN**：使用RNN处理部分可观测性
3. **IQL+通信**：引入通信机制
4. **混合方法**：IQL与值分解方法的结合

## 参考文献

- Tan, M. (1993). "Multi-agent reinforcement learning: Independent vs. cooperative agents." ICML.
- Tampuu, A., et al. (2017). "Multiagent cooperation and competition with deep reinforcement learning." PLoS ONE.

---

**注意**：IQL虽然简单，但在理解多智能体学习的基本原理和作为基准算法方面具有重要价值。在实际应用中，应根据任务特性选择合适的算法。