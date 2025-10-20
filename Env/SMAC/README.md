# SMAC (StarCraft Multi-Agent Challenge) 环境封装器

一个基于现有SMAC库(`smac.env.StarCraft2Env`)的封装器，提供与DEM、HRG、MSFS环境相同的统一接口。

## 概述

SMAC环境封装器是对标准SMAC库的封装，使其能够在统一的多智能体强化学习框架中使用。该封装器保留了原版SMAC的所有功能，同时提供了与其他环境一致的接口。

## 特性

- ✅ **统一接口**: 与DEM、HRG、MSFS环境具有相同的方法和返回格式
- ✅ **基于原版SMAC**: 使用经过验证的SMAC库作为底层引擎
- ✅ **CTDE兼容**: 支持QMIX、VDN、MADDPG等集中式训练分布式执行算法
- ✅ **多种地图**: 支持所有SMAC标准地图(8m、3s、2s3z、MMM、corridor等)
- ✅ **动作掩码**: 支持智能体动作可用性检测
- ✅ **配置灵活**: 支持多种难度和自定义配置
- ✅ **向后兼容**: 可以与原版SMAC代码无缝切换

## 支持的地图类型

| 地图名称 | 描述 | 智能体数量 | 特点 |
|---------|------|------------|------|
| 8m | 8个Marine vs 8个Marine | 8 | 对称战斗 |
| 3s | 3个Stalker vs 3个Zealot | 3 | 异构单位战斗 |
| 2s3z | 2个Stalker vs 3个Zealot | 5 | 数量不对称 |
| MMM | Marine、Medivac、Marauder组合 | 10 | 混合部队 |
| corridor | 狭窄走廊地图 | 6 | 限制移动空间 |
| 6h | 6个Hydralisk vs 多个单位 | 10+ | 大规模战斗 |
| 1c3s5z | 1个Colossus+3个Stalker vs 5个Zealot | 9 | 高级单位 |
| bane_vs_bane | Baneling vs Baneling | 20 | 自爆单位 |

## 安装要求

确保已安装SMAC库：
```bash
pip install smac
```

## 基础使用

### 创建环境

```python
from Env.SMAC import create_smac_env

# 创建基础环境
env = create_smac_env(map_name="8m", episode_limit=200)

# 重置环境
observations = env.reset()

# 执行步骤
actions = {}
for agent_id in env.agent_ids:
    avail_actions = env.get_avail_actions(agent_id)
    actions[agent_id] = np.random.choice(avail_actions) if avail_actions else 0

observations, rewards, dones, infos = env.step(actions)

# 关闭环境
env.close()
```

### CTDE模式

```python
from Env.SMAC import create_smac_ctde_env

# 创建CTDE环境
ctde_env = create_smac_ctde_env(
    map_name="3s",
    global_state_type="concat"
)

# 获取全局状态
global_state = ctde_env.get_global_state()

# 执行步骤
obs, rewards, dones, infos = ctde_env.step(actions)
global_state = infos['global_state']  # 全局状态在info中
```

### 不同难度配置

```python
from Env.SMAC import create_smac_env_easy, create_smac_env_normal, create_smac_env_hard

# 简单模式
easy_env = create_smac_env_easy()

# 普通模式
normal_env = create_smac_env_normal()

# 困难模式
hard_env = create_smac_env_hard()
```

## 接口对比

### 原版SMAC接口

```python
from smac.env import StarCraft2Env

env = StarCraft2Env(map_name="8m")
env.reset()
obs = env.get_obs()  # 返回列表
state = env.get_state()  # 返回numpy数组
reward, terminated, info = env.step(actions)  # 单个奖励
```

### 封装器接口

```python
from Env.SMAC import create_smac_env

env = create_smac_env(map_name="8m")
observations = env.reset()  # 返回字典
global_state = env.get_global_state()  # 全局状态
obs, rewards, dones, infos = env.step(actions)  # 每个智能体的奖励
```

### 统一接口优势

- **字典格式观察**: `{"agent_0": obs1, "agent_1": obs2, ...}`
- **分布式奖励**: 每个智能体独立的奖励
- **全局状态**: 支持CTDE算法的全局状态表示
- **动作掩码**: `get_avail_actions(agent_id)`方法
- **环境信息**: `get_env_info()`提供完整环境信息

## 观察和动作空间

### 观察空间

- **维度**: 80维（根据地图可能变化）
- **格式**: numpy数组，float32类型
- **包含**: 智能体自身状态、可见友军、可见敌人等信息

### 动作空间

- **类型**: 离散动作
- **数量**: 14个动作（根据地图可能变化）
- **动作掩码**: `get_avail_actions(agent_id)`返回可用动作列表

## 全局状态类型

CTDE环境支持4种全局状态表示：

- **concat**: 拼接所有智能体观察（n_agents × obs_dim）
- **mean**: 所有人观察的平均值（obs_dim）
- **max**: 所有人观察的最大值（obs_dim）
- **attention**: 基于注意力机制的重要性加权状态（obs_dim + min(5, n_agents)）

## 配置选项

主要配置参数：

```python
SMACConfig(
    map_name="8m",           # SMAC地图名称
    difficulty="normal",       # 难度级别
    episode_limit=None,        # 回合步数限制
    render_mode=None,          # 渲染模式
    debug=False,              # 调试模式
    seed=None                 # 随机种子
)
```

### 预定义配置

- **easy**: 简单地图，较长时间限制
- **normal**: 标准地图，标准时间限制
- **hard**: 复杂地图，较短时间限制
- **debug**: 调试配置，短时间限制

## 性能基准

| 地图 | 智能体数量 | 每秒步数 | 平均步时间 |
|------|------------|----------|------------|
| 3s | 3 | 10-50+ | <0.1秒 |
| 8m | 8 | 5-20+ | <0.2秒 |
| MMM | 10 | 2-10+ | <0.5秒 |

*注意：性能受StarCraft II游戏引擎影响，可能比自定义环境慢*

## 测试和验证

运行测试套件：

```bash
# 快速测试
python Env/SMAC/test_env.py --quick

# 完整测试
python Env/SMAC/test_env.py
```

运行教程：

```bash
jupyter notebooks tutorials/SMAC_Wrapper_Tutorial.ipynb
```

## 与其他环境的兼容性

SMAC封装器与DEM、HRG、MSFS环境具有相同的接口：

- `reset()`: 重置环境
- `step(actions)`: 执行步骤
- `get_observations()`: 获取观察
- `get_global_state()`: 获取全局状态
- `get_avail_actions(agent_id)`: 获取可用动作
- `get_env_info()`: 获取环境信息

这使得您可以在相同的算法框架中使用不同的环境。

## 迁移指南

### 从原版SMAC迁移

**原版代码**:
```python
from smac.env import StarCraft2Env

env = StarCraft2Env(map_name="8m")
env.reset()
obs = env.get_obs()
actions = [np.random.choice(env.get_avail_agent_actions(i))
              for i in range(env_info['n_agents'])]
reward, terminated, info = env.step(actions)
```

**封装器代码**:
```python
from Env.SMAC import create_smac_env

env = create_smac_env(map_name="8m")
observations = env.reset()
actions = {agent_id: np.random.choice(env.get_avail_actions(agent_id))
          for agent_id in env.agent_ids}
obs, rewards, dones, infos = env.step(actions)
```

### 主要变化

1. **观察格式**: 从列表变为字典，键为智能体ID
2. **奖励格式**: 从单个值变为每个智能体的字典
3. **动作格式**: 从列表变为字典，键为智能体ID
4. **返回值**: `step()`现在返回(obs, rewards, dones, infos)四元组

## 常见问题

### Q: 为什么需要封装器？
A: 为了在统一的多智能体强化学习框架中使用SMAC，使其与DEM、HRG、MSFS环境具有一致的接口。

### Q: 封装器会影响性能吗？
A: 封装器开销很小，主要性能瓶颈仍然是StarCraft II游戏引擎。

### Q: 如何使用自定义地图？
A: 直接在`create_smac_env()`或`SMACConfig`中指定`map_name`参数。

### Q: 支持哪些全局状态类型？
A: 支持concat、mean、max、attention四种类型，适合不同的CTDE算法。

## 贡献

欢迎提交问题报告和改进建议！

## 许可证

本项目遵循MIT许可证。SMAC库遵循其原始许可证。