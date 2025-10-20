# Multi-Agent Reinforcement Learning Environments

[中文文档](./README_CN.md) | English

## Overview

This repository provides a collection of **5 diverse multi-agent reinforcement learning (MARL) environments** designed for research and benchmarking of MARL algorithms, particularly Centralized Training with Decentralized Execution (CTDE) algorithms like QMIX, VDN, and MADDPG.

### Environment List

| Environment | Acronym | Task Type | Agents | Difficulty Levels |
|------------|---------|-----------|--------|-------------------|
| Collaborative Moving | CM | Cooperation & Coordination | 2-4 | debug, easy, normal, hard |
| Dynamic Escort Mission | DEM | Dynamic Role Formation | 3 | easy, normal, hard |
| Heterogeneous Resource Gathering | HRG | Heterogeneous Cooperation | 2-6 | easy, normal, hard, ultra_fast |
| Smart Manufacturing Flow Scheduling | MSFS | Role Emergence | 1-3 | easy, normal, hard |
| StarCraft Multi-Agent Challenge | SMAC | Combat Strategy | map-dependent | easy, normal, hard |

---

## 1. CM (Collaborative Moving) Environment

### Task Description

Agents must cooperate to push a box from its initial position to a target location. The box can only be moved successfully when multiple agents push from different sides, with success probability increasing with the number of cooperating agents.

### Key Features
- **Cooperation Mechanism**: Box pushing requires coordination from multiple sides
- **Probabilistic Success**: 
  - 1 agent: 50% success rate
  - 2 agents: 75% success rate
  - 3 agents: 90% success rate
  - 4 agents: 100% success rate
- **Configurable Difficulty**: 4 difficulty levels with varying grid sizes and agent counts

### Action Space (5 discrete actions per agent)

| Action ID | Name | Description |
|-----------|------|-------------|
| 0 | STAY | Agent remains in current position |
| 1 | UP | Move one grid cell upward |
| 2 | DOWN | Move one grid cell downward |
| 3 | LEFT | Move one grid cell left |
| 4 | RIGHT | Move one grid cell right |

### Observation Space

**Vector format (length = 6 + 2×(n_agents-1)):**
- Agent's own position (2 values)
- Box center position (2 values)
- Goal center position (2 values)
- Relative positions of other agents (2×(n_agents-1) values)

For 2-agent environment, observation length = 8:
```
[self_x, self_y, box_x, box_y, goal_x, goal_y, other_rel_x, other_rel_y]
```

### CTDE Global State

**Global State Components:**
- All agent positions (2 × n_agents values)
- Box position and size (3 values)
- Goal position and size (3 values)
- Relative positions between agents

**Global State Types:**
- `concat`: Concatenation of all information (default)
- `mean`: Mean pooling of agent observations
- `max`: Max pooling of agent observations
- `attention`: Attention-based aggregation

### Reward System

- **Time Penalty**: -0.3 per step (encourages efficiency)
- **Distance Improvement**: 0.3 × distance reduction
- **Box Movement**: 1.0 (when box moves toward goal)
- **Cooperation Bonus**: 1.5 × (n_pushing_agents - 1)
- **Goal Completion**: 50.0 + efficiency bonus (up to +15.0)

**Reward Span**: ~80 units (from random exploration to goal completion)

### Difficulty Levels

| Difficulty | Grid Size | Agents | Max Steps | Goal Reward | Success Probs |
|------------|-----------|--------|-----------|-------------|---------------|
| debug | 5×5 | 2 | 50 | 20.0 | {1: 0.8, 2: 1.0} |
| easy | 7×7 | 2 | 100 | 80.0 | {1: 0.7, 2: 0.9} |
| normal | 7×7 | 2 | 100 | 50.0 | {1: 0.5, 2: 0.75, 3: 0.9} |
| hard | 9×9 | 3 | 150 | 100.0 | {1: 0.3, 2: 0.6, 3: 0.85} |

### Usage Example

```python
from Env.CM.env_cm import create_cm_env
from Env.CM.env_cm_ctde import create_cm_ctde_env

# Create standard environment
env = create_cm_env(difficulty="normal", render_mode="rgb_array")

# Create CTDE environment
ctde_env = create_cm_ctde_env(
    difficulty="normal_ctde", 
    global_state_type="concat"
)

# Reset environment
obs = env.reset()

# Step
actions = {agent_id: env.get_avail_actions(agent_id)[0] 
           for agent_id in env.agent_ids}
obs, rewards, dones, info = env.step(actions)

# Get global state (CTDE only)
global_state = ctde_env.get_global_state()
```

---

## 2. DEM (Dynamic Escort Mission) Environment

### Task Description

Special forces agents must escort a VIP through dangerous territory while dynamically forming roles (Guard, Vanguard, Sniper) to deal with various threats. The VIP moves autonomously using intelligent pathfinding, and agents must protect it while clearing threats.

### Key Features
- **Dynamic Role Formation**: Agents naturally form roles through reward shaping
- **Intelligent VIP**: Autonomous pathfinding with obstacle avoidance
- **Diverse Threats**: Rushers (fast, melee) and Shooters (long-range, stationary)
- **Terrain Types**: Rivers (impassable), Forests (damage reduction)

### Action Space (10 discrete actions per agent)

| Action ID | Name | Description |
|-----------|------|-------------|
| 0 | STAY | No action |
| 1-4 | MOVE | Move up/down/left/right |
| 5 | ATTACK | Attack nearest threat in range |
| 6 | GUARD_VIP | Guard VIP (reduces incoming damage) |
| 7 | WARN_THREAT | Send threat warning message |
| 8 | ALL_CLEAR | Send all-clear message |
| 9 | OBSERVE | Observe environment |

### Observation Space (59 dimensions)

**Self State (8 dimensions):**
- Position (2), HP (1), Attack cooldown (1)
- Guarding status (1), Distance to VIP (1), Distance to target (1), In forest (1)

**VIP State (6 dimensions):**
- Visible (1), HP (1), Relative position (2), Under attack (1), Adjacent (1)

**Teammates (12 dimensions):**
- Up to 2 teammates × 6 dimensions (relative pos, HP, adjacent to VIP, guarding, cooldown)

**Threats (25 dimensions):**
- Up to 5 threats × 5 dimensions (type, relative pos, HP, cooldown)

**Communication (6 dimensions):**
- 3 recent messages × 2 dimensions (type, age)

**Additional Info (2 dimensions):**
- Normalized step, constant bias

### Reward System

**Main Rewards:**
- VIP reaches target: +50.0
- VIP death: -30.0
- VIP progress: +0.2 per grid unit closer
- Threat killed: +3.0
- Long-range kill (≥6 units): +1.0

**Role Emergence Rewards:**
- Guard adjacent to VIP: +0.05
- Body block (damage reduction): +0.5
- Vanguard ahead of VIP: +0.05
- Good spread (avg distance 2-5): +0.02

**Penalties:**
- VIP damage: -0.1 per HP
- Agent death: -3.0
- Collision: -0.05
- Invalid action: -0.1

### Difficulty Levels

| Difficulty | Grid Size | VIP HP | Agent HP | Threats | Spawn Interval |
|------------|-----------|--------|----------|---------|----------------|
| easy | 10×10 | 80 | 60 | max 3 | 10-12 steps |
| normal | 12×12 | 60 | 50 | max 5 | 6-8 steps |
| hard | 12×12 | 40 | 40 | max 8 | 4-6 steps |

### Usage Example

```python
from Env.DEM.env_dem import DEMEnv
from Env.DEM.env_dem_ctde import DEMCTDEEnv
from Env.DEM.config import DEMConfig

# Create environment with custom config
config = DEMConfig(difficulty="normal")
env = DEMEnv(config)

# Or use CTDE wrapper
ctde_env = DEMCTDEEnv(difficulty="normal")

# Reset and step
obs = env.reset()
actions = {agent_id: 0 for agent_id in env.game_state.agents.keys()}
obs, rewards, dones, info = env.step(actions)
```

---

## 3. HRG (Heterogeneous Resource Gathering) Environment

### Task Description

Agents with different roles (Scouts, Workers, Transporters) work together to collect resources (Gold, Wood) and bring them back to base. Each agent type has unique capabilities:
- **Scouts**: High vision, fast movement, cannot gather
- **Workers**: Can gather resources, medium capacity
- **Transporters**: High carrying capacity, fast movement

### Key Features
- **Heterogeneous Agents**: 3 distinct agent types with specialized abilities
- **Resource Types**: Gold (high value, clustered) and Wood (lower value, distributed)
- **Role-based Cooperation**: Efficient resource collection requires coordination

### Action Space (8 discrete actions per agent)

| Action ID | Name | Description |
|-----------|------|-------------|
| 0-3 | MOVE | Move North/South/West/East |
| 4 | GATHER | Gather resource at current position (Workers only) |
| 5 | TRANSFER | Transfer resources to adjacent agent |
| 6 | DEPOSIT | Deposit resources at base |
| 7 | WAIT | Wait (no action) |

### Observation Space (60 dimensions - optimized)

**Agent Self State (10 dimensions):**
- Position (2), Role (one-hot, 3), Inventory (2), Energy (1), Cooldown (1), Distance to base (1), Time remaining (1)

**Visible Entities (40 dimensions):**
- Up to 6 entities within vision range
- Each entity: relative position (2) + type info (3) = 5 dimensions

**Communication (10 dimensions):**
- 3 recent messages × ~3 dimensions

### Global State (120 dimensions - optimized)

- Agent states: 6 agents × 12 dimensions = 72
- Resource summary: 24 dimensions (clustered by quadrant)
- Global statistics: 24 dimensions

### Reward System

**Resource Values:**
- Gold: 10.0 per unit
- Wood: 2.0 per unit

**Action Rewards:**
- Gather: 10% of resource value
- Transfer: 5% of resource value
- Deposit: 50% of resource value

**Team Rewards:**
- Time penalty: -0.01 per step
- Resource diversity bonus: +0.1 for gold, +0.05 for wood

### Difficulty Levels

| Difficulty | Grid Size | Max Steps | Gold | Wood | Obstacles | Agents |
|------------|-----------|-----------|------|------|-----------|--------|
| easy | 8×8 | 300 | 2 | 15 | 0 | 6 (2S, 3W, 1T) |
| normal | 10×10 | 200 | 3 | 10 | 10 | 6 (2S, 3W, 1T) |
| hard | 12×12 | 150 | 4 | 8 | 20 | 6 (2S, 3W, 1T) |
| ultra_fast | 6×6 | 80 | 1 | 4 | 2 | 2 (1W, 1T) |

S=Scout, W=Worker, T=Transporter

### Usage Example

```python
from Env.HRG.env_hrg import create_hrg_env
from Env.HRG.env_hrg_ctde import create_hrg_ctde_env

# Create environment
env = create_hrg_env(difficulty="normal")

# Or use ultra-fast version for training
fast_env = create_hrg_env(difficulty="ultra_fast")

# CTDE version
ctde_env = create_hrg_ctde_env(difficulty="normal")

# Reset and step
obs = env.reset()
global_state = ctde_env.get_global_state()
```

---

## 4. MSFS (Smart Manufacturing Flow Scheduling) Environment

### Task Description

Robots must collaboratively process orders through a 3-stage manufacturing pipeline (Raw → Assembly → Packing). Agents naturally form specialized roles through reward shaping, acting as Collectors, Processors, or Packagers.

### Key Features
- **Role Emergence**: Agents specialize through consecutive processing at same station
- **Order Types**: Simple (fast, low value) and Complex (slow, high value)
- **3-Stage Pipeline**: Each order must pass through all three workstations
- **Dynamic Queue Management**: Agents must balance workloads

### Action Space (8 discrete actions per agent)

| Action ID | Name | Description |
|-----------|------|-------------|
| 0 | WAIT | No action |
| 1-3 | MOVE_TO_STATION | Move to Raw/Assembly/Packing station |
| 4 | PULL_ORDER | Pull order from queue (Raw station only) |
| 5 | START_PROCESSING | Begin or continue processing |
| 6 | COMPLETE_STAGE | Complete current stage, move to next |
| 7 | DELIVER_ORDER | Deliver finished order (Packing station only) |

### Observation Space (24 dimensions)

**Self State (10 dimensions):**
- Current workstation (one-hot, 3)
- Move cooldown (1), Carrying status (1)
- Order type & stage info (5)

**Global Info (7 dimensions):**
- Queue lengths (3), Order counts (2), Time (2)

**Teammate Info (7 dimensions):**
- Teammate workstation (one-hot, 3)
- Busy status (1), Carrying/processing info (3)

### Global State (42 dimensions)

- Agent states: 2 agents × 8 dimensions = 16
- Workstation states: 3 stations × 6 dimensions = 18
- Global statistics: 8 dimensions

### Reward System (Enhanced for Exploration)

**Action-based Rewards (Immediate):**
- Move toward target: +0.1
- Pickup material: +0.2
- Start processing: +0.3
- Complete stage: +0.5
- Deliver order: +1.0

**Progress Rewards (Milestone-based):**
- Raw completion: +1.0
- Assembly completion: +2.0
- Packaging completion: +3.0
- Order delivery: +5.0
- Smooth workflow bonus: +0.5

**Cooperation Rewards:**
- Successful handoff: +0.8
- Workstation ready: +0.4
- Concurrent processing: +0.6
- Balanced workload: +0.3

**Role Emergence Rewards:**
- Collector/Processor/Packager focus: +0.2/0.3/0.4
- Stick to role: +0.1
- Adaptive switching: +0.5

**Penalties (Light):**
- Invalid action: -0.1

### Difficulty Levels

| Difficulty | Max Steps | Order Values | Reward Scale | Penalties |
|------------|-----------|--------------|--------------|-----------|
| easy | 60 | Simple: 7.0, Complex: 12.0 | 1.5× | Minimal |
| normal | 50 | Simple: 5.0, Complex: 10.0 | 1.0× | Standard |
| hard | 40 | Simple: 4.0, Complex: 8.0 | 0.7× | Higher |

### Usage Example

```python
from Env.MSFS.env_msfs import create_msfs_env
from Env.MSFS.env_msfs_ctde import create_msfs_ctde_env

# Create environment
env = create_msfs_env(difficulty="normal")

# CTDE version
ctde_env = create_msfs_ctde_env(difficulty="normal")

# Reset and step
obs = env.reset()
actions = {agent_id: 0 for agent_id in env.game_state.agents.keys()}
obs, rewards, dones, info = env.step(actions)
```

---

## 5. SMAC (StarCraft Multi-Agent Challenge) Wrapper

### Task Description

A wrapper for the StarCraft Multi-Agent Challenge environment, providing a standardized interface compatible with our MARL framework. Agents control units in StarCraft II to defeat enemy forces.

### Key Features
- **Official SMAC Maps**: Support for all official SMAC scenarios
- **Standardized Interface**: Compatible with QMIX, VDN, and other CTDE algorithms
- **Multiple Scenarios**: From simple (3m, 8m) to complex (MMM, corridor)

### Map Categories

**Homogeneous Units:**
- `2m`, `3m`, `4m`, `5m`, `8m`, `10m` - Marine units
- `2s`, `3s`, `4s`, `5s` - Stalker units

**Heterogeneous Units:**
- `2s3z`, `3s5z`, `1c3s5z` - Mixed unit types
- `MMM`, `MMM2` - Marines, Marauders, Medivacs

**Asymmetric Scenarios:**
- `2m_vs_1z`, `3s_vs_5z`, `2c_vs_64zg` - Imbalanced battles

**Complex Scenarios:**
- `corridor` - Narrow passage combat
- `6h_vs_8z` - Hellions vs Zerglings

### Action Space

- Number of actions varies by scenario (typically 6-20)
- Includes: no-op, stop, move directions, attack enemy units

### Observation Space

- Local observations per agent (varies by scenario, typically 40-100 dimensions)
- Includes: own unit features, enemy features, ally features, terrain

### Global State

- Full game state including all unit positions, HP, shields, etc.
- Dimension varies by scenario (typically 100-300 dimensions)

### Usage Example

```python
from Env.SMAC.env_smac import SMACEnv
from Env.SMAC.env_smac_ctde import SMACCTDEEnv

# Create standard environment
env = SMACEnv(map_name="8m")

# Create CTDE environment
ctde_env = SMACCTDEEnv(map_name="8m")

# Reset and step
obs = env.reset()
actions = {agent_id: 0 for agent_id in env.agent_ids}
obs, rewards, dones, info = env.step(actions)

# Get global state
global_state = ctde_env.get_global_state()
```

**Note**: SMAC requires StarCraft II installation. See [SMAC README](./SMAC/README.md) for installation instructions.

---

## Installation

### Basic Requirements

```bash
# Ubuntu 24.04, Python 3.8+
pip install numpy gymnasium matplotlib
```

### Optional Requirements

For SMAC environment:
```bash
pip install -r Env/doc/requirements_with_smac.txt
```

See [requirements.txt](./doc/requirements.txt) for full dependency list.

---

## Environment Compatibility

All environments are compatible with:
- **QMIX**: Centralized value factorization
- **VDN**: Value Decomposition Networks
- **MADDPG**: Multi-Agent DDPG
- **Other CTDE algorithms**: Through standardized interfaces

### Standard Interface

All environments provide:
```python
# Reset
observations = env.reset()

# Step
observations, rewards, dones, info = env.step(actions)

# Get environment info
env_info = env.get_env_info()
# Returns: n_agents, agent_ids, n_actions, obs_dims, act_dims, episode_limit

# Get available actions (for action masking)
avail_actions = env.get_avail_actions(agent_id)

# Get global state (CTDE environments)
global_state = ctde_env.get_global_state()
```

---

## Tutorials

Interactive Jupyter tutorials are provided for each environment:
- [CM Environment Tutorial](./CM_environment_tutorial.ipynb)
- [DEM Environment Tutorial](./DEM_environment_tutorial.ipynb)
- [HRG Environment Tutorial](./HRG_environment_tutorial.ipynb)
- [MSFS Environment Tutorial](./MSFS_environment_tutorial.ipynb)
- [SMAC Wrapper Tutorial](./SMAC_environment_tutorial.ipynb)

---

## Citation

If you use these environments in your research, please cite:

```bibtex
@misc{marl_envs_2024,
  title={Multi-Agent Reinforcement Learning Environments},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/MARL}
}
```

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Contact

For questions, issues, or contributions, please open an issue on GitHub or contact the maintainers.
