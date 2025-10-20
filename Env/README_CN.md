# 多智能体强化学习环境集合

中文文档 | [English](./README.md)

## 概述

本仓库提供了**5个多样化的多智能体强化学习(MARL)环境**,专为研究和评估MARL算法而设计,特别适用于集中训练分散执行(CTDE)算法,如QMIX、VDN和MADDPG。

### 环境列表

| 环境名称 | 缩写 | 任务类型 | 智能体数量 | 难度等级 |
|---------|------|---------|-----------|---------|
| 协作推箱 | CM | 合作与协调 | 2-4 | debug, easy, normal, hard |
| 动态护送任务 | DEM | 动态角色形成 | 3 | easy, normal, hard |
| 异构资源采集 | HRG | 异构合作 | 2-6 | easy, normal, hard, ultra_fast |
| 智能制造流程调度 | MSFS | 角色涌现 | 1-3 | easy, normal, hard |
| 星际争霸多智能体挑战 | SMAC | 战斗策略 | 地图决定 | easy, normal, hard |

---

## 1. CM (协作推箱) 环境

### 任务描述

智能体必须合作将一个箱子从初始位置推到目标位置。箱子只有在多个智能体从不同方向推动时才能成功移动,成功概率随合作智能体数量增加而提高。

### 核心特性
- **合作机制**: 推箱需要多方向协调
- **概率成功**: 
  - 1个智能体: 50%成功率
  - 2个智能体: 75%成功率
  - 3个智能体: 90%成功率
  - 4个智能体: 100%成功率
- **可配置难度**: 4个难度级别,网格大小和智能体数量各不相同

### 动作空间 (每个智能体5个离散动作)

| 动作ID | 名称 | 描述 |
|-------|------|------|
| 0 | 保持 | 智能体保持当前位置 |
| 1 | 上移 | 向上移动一格 |
| 2 | 下移 | 向下移动一格 |
| 3 | 左移 | 向左移动一格 |
| 4 | 右移 | 向右移动一格 |

### 观测空间

**向量格式 (长度 = 6 + 2×(n_agents-1)):**
- 智能体自身位置 (2个值)
- 箱子中心位置 (2个值)
- 目标中心位置 (2个值)
- 其他智能体的相对位置 (2×(n_agents-1)个值)

对于2智能体环境,观测长度 = 8:
```
[self_x, self_y, box_x, box_y, goal_x, goal_y, other_rel_x, other_rel_y]
```

### CTDE全局状态

**全局状态组成:**
- 所有智能体位置 (2 × n_agents 个值)
- 箱子位置和大小 (3个值)
- 目标位置和大小 (3个值)
- 智能体间相对位置

**全局状态类型:**
- `concat`: 拼接所有信息 (默认)
- `mean`: 智能体观测的平均池化
- `max`: 智能体观测的最大池化
- `attention`: 基于注意力的聚合

### 奖励系统

- **时间惩罚**: 每步-0.3 (鼓励高效)
- **距离改善**: 0.3 × 距离减少量
- **箱子移动**: 1.0 (当箱子向目标移动时)
- **合作奖励**: 1.5 × (推箱智能体数 - 1)
- **目标完成**: 50.0 + 效率奖励(最高+15.0)

**奖励跨度**: ~80单位 (从随机探索到目标完成)

### 难度等级

| 难度 | 网格大小 | 智能体 | 最大步数 | 目标奖励 | 成功概率 |
|-----|---------|--------|---------|---------|---------|
| debug | 5×5 | 2 | 50 | 20.0 | {1: 0.8, 2: 1.0} |
| easy | 7×7 | 2 | 100 | 80.0 | {1: 0.7, 2: 0.9} |
| normal | 7×7 | 2 | 100 | 50.0 | {1: 0.5, 2: 0.75, 3: 0.9} |
| hard | 9×9 | 3 | 150 | 100.0 | {1: 0.3, 2: 0.6, 3: 0.85} |

### 使用示例

```python
from Env.CM.env_cm import create_cm_env
from Env.CM.env_cm_ctde import create_cm_ctde_env

# 创建标准环境
env = create_cm_env(difficulty="normal", render_mode="rgb_array")

# 创建CTDE环境
ctde_env = create_cm_ctde_env(
    difficulty="normal_ctde", 
    global_state_type="concat"
)

# 重置环境
obs = env.reset()

# 执行步骤
actions = {agent_id: env.get_avail_actions(agent_id)[0] 
           for agent_id in env.agent_ids}
obs, rewards, dones, info = env.step(actions)

# 获取全局状态 (仅CTDE)
global_state = ctde_env.get_global_state()
```

---

## 2. DEM (动态护送任务) 环境

### 任务描述

特种部队智能体必须护送VIP穿越危险地带,同时动态形成角色(护卫、先锋、狙击手)应对各种威胁。VIP使用智能寻路自主移动,智能体必须保护它并清除威胁。

### 核心特性
- **动态角色形成**: 智能体通过奖励塑造自然形成角色
- **智能VIP**: 具有障碍物规避的自主寻路
- **多样威胁**: 冲锋者(快速,近战)和射手(远程,固定)
- **地形类型**: 河流(不可通过)、森林(伤害减免)

### 动作空间 (每个智能体10个离散动作)

| 动作ID | 名称 | 描述 |
|-------|------|------|
| 0 | 等待 | 无动作 |
| 1-4 | 移动 | 向上/下/左/右移动 |
| 5 | 攻击 | 攻击范围内最近的威胁 |
| 6 | 保护VIP | 保护VIP(减少受到的伤害) |
| 7 | 威胁警告 | 发送威胁警告消息 |
| 8 | 全部清除 | 发送全部清除消息 |
| 9 | 观察 | 观察环境 |

### 观测空间 (59维)

**自身状态 (8维):**
- 位置(2), 生命值(1), 攻击冷却(1)
- 保护状态(1), 到VIP距离(1), 到目标距离(1), 在森林中(1)

**VIP状态 (6维):**
- 可见(1), 生命值(1), 相对位置(2), 受攻击(1), 相邻(1)

**队友 (12维):**
- 最多2个队友 × 6维 (相对位置, 生命值, VIP相邻, 保护中, 冷却)

**威胁 (25维):**
- 最多5个威胁 × 5维 (类型, 相对位置, 生命值, 冷却)

**通信 (6维):**
- 3条最近消息 × 2维 (类型, 年龄)

**额外信息 (2维):**
- 归一化步数, 常数偏置

### 奖励系统

**主要奖励:**
- VIP到达目标: +50.0
- VIP死亡: -30.0
- VIP前进: 每格+0.2
- 击杀威胁: +3.0
- 远程击杀(≥6格): +1.0

**角色涌现奖励:**
- 护卫在VIP附近: +0.05
- 身体阻挡(伤害减免): +0.5
- 先锋在VIP前方: +0.05
- 良好分散(平均距离2-5): +0.02

**惩罚:**
- VIP受伤: 每点生命值-0.1
- 智能体死亡: -3.0
- 碰撞: -0.05
- 无效动作: -0.1

### 难度等级

| 难度 | 网格大小 | VIP生命值 | 智能体生命值 | 威胁 | 生成间隔 |
|-----|---------|----------|------------|------|---------|
| easy | 10×10 | 80 | 60 | 最多3个 | 10-12步 |
| normal | 12×12 | 60 | 50 | 最多5个 | 6-8步 |
| hard | 12×12 | 40 | 40 | 最多8个 | 4-6步 |

### 使用示例

```python
from Env.DEM.env_dem import DEMEnv
from Env.DEM.env_dem_ctde import DEMCTDEEnv
from Env.DEM.config import DEMConfig

# 使用自定义配置创建环境
config = DEMConfig(difficulty="normal")
env = DEMEnv(config)

# 或使用CTDE包装器
ctde_env = DEMCTDEEnv(difficulty="normal")

# 重置和步进
obs = env.reset()
actions = {agent_id: 0 for agent_id in env.game_state.agents.keys()}
obs, rewards, dones, info = env.step(actions)
```

---

## 3. HRG (异构资源采集) 环境

### 任务描述

具有不同角色的智能体(侦察兵、工人、运输兵)协作收集资源(金矿、木材)并运回基地。每种智能体类型具有独特能力:
- **侦察兵**: 高视野,快速移动,不能采集
- **工人**: 可采集资源,中等容量
- **运输兵**: 高携带容量,快速移动

### 核心特性
- **异构智能体**: 3种不同智能体类型,各有专长
- **资源类型**: 金矿(高价值,聚集)和木材(低价值,分散)
- **基于角色的合作**: 高效资源收集需要协调

### 动作空间 (每个智能体8个离散动作)

| 动作ID | 名称 | 描述 |
|-------|------|------|
| 0-3 | 移动 | 向北/南/西/东移动 |
| 4 | 采集 | 在当前位置采集资源(仅工人) |
| 5 | 转移 | 向相邻智能体转移资源 |
| 6 | 存放 | 在基地存放资源 |
| 7 | 等待 | 等待(无动作) |

### 观测空间 (60维 - 已优化)

**智能体自身状态 (10维):**
- 位置(2), 角色(独热,3), 库存(2), 能量(1), 冷却(1), 到基地距离(1), 剩余时间(1)

**可见实体 (40维):**
- 视野范围内最多6个实体
- 每个实体: 相对位置(2) + 类型信息(3) = 5维

**通信 (10维):**
- 3条最近消息 × ~3维

### 全局状态 (120维 - 已优化)

- 智能体状态: 6个智能体 × 12维 = 72
- 资源摘要: 24维(按象限聚类)
- 全局统计: 24维

### 奖励系统

**资源价值:**
- 金矿: 每单位10.0
- 木材: 每单位2.0

**动作奖励:**
- 采集: 资源价值的10%
- 转移: 资源价值的5%
- 存放: 资源价值的50%

**团队奖励:**
- 时间惩罚: 每步-0.01
- 资源多样性奖励: 金矿+0.1,木材+0.05

### 难度等级

| 难度 | 网格大小 | 最大步数 | 金矿 | 木材 | 障碍物 | 智能体 |
|-----|---------|---------|------|------|-------|--------|
| easy | 8×8 | 300 | 2 | 15 | 0 | 6 (2S, 3W, 1T) |
| normal | 10×10 | 200 | 3 | 10 | 10 | 6 (2S, 3W, 1T) |
| hard | 12×12 | 150 | 4 | 8 | 20 | 6 (2S, 3W, 1T) |
| ultra_fast | 6×6 | 80 | 1 | 4 | 2 | 2 (1W, 1T) |

S=侦察兵, W=工人, T=运输兵

### 使用示例

```python
from Env.HRG.env_hrg import create_hrg_env
from Env.HRG.env_hrg_ctde import create_hrg_ctde_env

# 创建环境
env = create_hrg_env(difficulty="normal")

# 或使用超快版本进行训练
fast_env = create_hrg_env(difficulty="ultra_fast")

# CTDE版本
ctde_env = create_hrg_ctde_env(difficulty="normal")

# 重置和步进
obs = env.reset()
global_state = ctde_env.get_global_state()
```

---

## 4. MSFS (智能制造流程调度) 环境

### 任务描述

机器人必须协作处理订单,经过3阶段制造流程(原料 → 组装 → 包装)。智能体通过奖励塑造自然形成专业化角色,充当收集者、处理者或包装者。

### 核心特性
- **角色涌现**: 智能体通过在同一工位连续处理而专业化
- **订单类型**: 简单(快速,低价值)和复杂(慢速,高价值)
- **3阶段流水线**: 每个订单必须通过所有三个工位
- **动态队列管理**: 智能体必须平衡工作负载

### 动作空间 (每个智能体8个离散动作)

| 动作ID | 名称 | 描述 |
|-------|------|------|
| 0 | 等待 | 无动作 |
| 1-3 | 移动到工位 | 移动到原料/组装/包装工位 |
| 4 | 拉取订单 | 从队列拉取订单(仅原料工位) |
| 5 | 开始处理 | 开始或继续处理 |
| 6 | 完成阶段 | 完成当前阶段,移至下一阶段 |
| 7 | 交付订单 | 交付完成的订单(仅包装工位) |

### 观测空间 (24维)

**自身状态 (10维):**
- 当前工位(独热,3)
- 移动冷却(1), 携带状态(1)
- 订单类型和阶段信息(5)

**全局信息 (7维):**
- 队列长度(3), 订单计数(2), 时间(2)

**队友信息 (7维):**
- 队友工位(独热,3)
- 忙碌状态(1), 携带/处理信息(3)

### 全局状态 (42维)

- 智能体状态: 2个智能体 × 8维 = 16
- 工位状态: 3个工位 × 6维 = 18
- 全局统计: 8维

### 奖励系统 (增强探索版)

**基于动作的奖励 (即时):**
- 向目标移动: +0.1
- 拾取材料: +0.2
- 开始处理: +0.3
- 完成阶段: +0.5
- 交付订单: +1.0

**进度奖励 (里程碑式):**
- 原料完成: +1.0
- 组装完成: +2.0
- 包装完成: +3.0
- 订单交付: +5.0
- 流畅工作流奖励: +0.5

**合作奖励:**
- 成功交接: +0.8
- 工位就绪: +0.4
- 并发处理: +0.6
- 负载均衡: +0.3

**角色涌现奖励:**
- 收集者/处理者/包装者专注: +0.2/0.3/0.4
- 坚持角色: +0.1
- 自适应切换: +0.5

**惩罚 (轻量):**
- 无效动作: -0.1

### 难度等级

| 难度 | 最大步数 | 订单价值 | 奖励规模 | 惩罚 |
|-----|---------|---------|---------|------|
| easy | 60 | 简单:7.0, 复杂:12.0 | 1.5× | 最小 |
| normal | 50 | 简单:5.0, 复杂:10.0 | 1.0× | 标准 |
| hard | 40 | 简单:4.0, 复杂:8.0 | 0.7× | 较高 |

### 使用示例

```python
from Env.MSFS.env_msfs import create_msfs_env
from Env.MSFS.env_msfs_ctde import create_msfs_ctde_env

# 创建环境
env = create_msfs_env(difficulty="normal")

# CTDE版本
ctde_env = create_msfs_ctde_env(difficulty="normal")

# 重置和步进
obs = env.reset()
actions = {agent_id: 0 for agent_id in env.game_state.agents.keys()}
obs, rewards, dones, info = env.step(actions)
```

---

## 5. SMAC (星际争霸多智能体挑战) 包装器

### 任务描述

StarCraft多智能体挑战环境的包装器,提供与我们MARL框架兼容的标准化接口。智能体控制星际争霸II中的单位击败敌军。

### 核心特性
- **官方SMAC地图**: 支持所有官方SMAC场景
- **标准化接口**: 与QMIX、VDN等CTDE算法兼容
- **多种场景**: 从简单(3m, 8m)到复杂(MMM, corridor)

### 地图类别

**同质单位:**
- `2m`, `3m`, `4m`, `5m`, `8m`, `10m` - 机枪兵单位
- `2s`, `3s`, `4s`, `5s` - 追猎者单位

**异质单位:**
- `2s3z`, `3s5z`, `1c3s5z` - 混合单位类型
- `MMM`, `MMM2` - 机枪兵、掠夺者、医疗艇

**非对称场景:**
- `2m_vs_1z`, `3s_vs_5z`, `2c_vs_64zg` - 不平衡战斗

**复杂场景:**
- `corridor` - 狭窄通道战斗
- `6h_vs_8z` - 地狱火兵对跳虫

### 动作空间

- 动作数量因场景而异(通常6-20个)
- 包括: 无操作、停止、移动方向、攻击敌方单位

### 观测空间

- 每个智能体的局部观测(因场景而异,通常40-100维)
- 包括: 己方单位特征、敌方特征、友军特征、地形

### 全局状态

- 完整游戏状态,包括所有单位位置、生命值、护盾等
- 维度因场景而异(通常100-300维)

### 使用示例

```python
from Env.SMAC.env_smac import SMACEnv
from Env.SMAC.env_smac_ctde import SMACCTDEEnv

# 创建标准环境
env = SMACEnv(map_name="8m")

# 创建CTDE环境
ctde_env = SMACCTDEEnv(map_name="8m")

# 重置和步进
obs = env.reset()
actions = {agent_id: 0 for agent_id in env.agent_ids}
obs, rewards, dones, info = env.step(actions)

# 获取全局状态
global_state = ctde_env.get_global_state()
```

**注意**: SMAC需要安装StarCraft II。请参阅 [SMAC README](./SMAC/README.md) 了解安装说明。

---

## 安装

### 基本要求

```bash
# Ubuntu 24.04, Python 3.8+
pip install numpy gymnasium matplotlib
```

### 可选要求

对于SMAC环境:
```bash
pip install -r Env/doc/requirements_with_smac.txt
```

完整依赖列表请参见 [requirements.txt](./doc/requirements.txt)。

---

## 环境兼容性

所有环境兼容:
- **QMIX**: 集中式价值分解
- **VDN**: 价值分解网络
- **MADDPG**: 多智能体DDPG
- **其他CTDE算法**: 通过标准化接口

### 标准接口

所有环境提供:
```python
# 重置
observations = env.reset()

# 步进
observations, rewards, dones, info = env.step(actions)

# 获取环境信息
env_info = env.get_env_info()
# 返回: n_agents, agent_ids, n_actions, obs_dims, act_dims, episode_limit

# 获取可用动作 (用于动作掩码)
avail_actions = env.get_avail_actions(agent_id)

# 获取全局状态 (CTDE环境)
global_state = ctde_env.get_global_state()
```

---

## 教程

为每个环境提供了交互式Jupyter教程:
- [CM环境教程](./CM_environment_tutorial.ipynb)
- [DEM环境教程](./DEM_environment_tutorial.ipynb)
- [HRG环境教程](./HRG_environment_tutorial.ipynb)
- [MSFS环境教程](./MSFS_environment_tutorial.ipynb)
- [SMAC包装器教程](./SMAC_environment_tutorial.ipynb)

---

## 引用

如果您在研究中使用这些环境,请引用:

```bibtex
@misc{marl_envs_2024,
  title={Multi-Agent Reinforcement Learning Environments},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/MARL}
}
```

---

## 许可证

本项目采用MIT许可证 - 详情请参见LICENSE文件。

---

## 联系方式

如有问题、议题或贡献,请在GitHub上提交issue或联系维护者。
