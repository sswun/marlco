"""
DEM Environment Core Components

This module defines the core data structures and enums for the DEM environment.
"""

from enum import Enum, IntEnum
from typing import NamedTuple, List, Optional, Dict, Any
import numpy as np


class TerrainType(Enum):
    """Terrain types in the environment"""
    OPEN = "open"
    FOREST = "forest"  # Provides damage reduction
    RIVER = "river"    # Impassable


class ThreatType(Enum):
    """Types of threats"""
    RUSHER = "rusher"   # Melee unit
    SHOOTER = "shooter" # Ranged unit


class ActionType(IntEnum):
    """Action types for agents"""
    STAY = 0
    MOVE_UP = 1
    MOVE_DOWN = 2
    MOVE_LEFT = 3
    MOVE_RIGHT = 4
    ATTACK = 5
    GUARD_VIP = 6
    WARN_THREAT = 7
    ALL_CLEAR = 8
    OBSERVE = 9


class MessageType(IntEnum):
    """Message types for communication"""
    THREAT_WARNING = 1
    ALL_CLEAR = 2


class Position(NamedTuple):
    """Position on the grid"""
    x: int
    y: int

    def manhattan_distance(self, other: 'Position') -> int:
        """Calculate Manhattan distance to another position"""
        return abs(self.x - other.x) + abs(self.y - other.y)

    def __add__(self, other: 'Position') -> 'Position':
        """Add two positions"""
        return Position(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Position') -> 'Position':
        """Subtract two positions"""
        return Position(self.x - other.x, self.y - other.y)


# Direction vectors for movement
DIRECTIONS = {
    ActionType.MOVE_UP: Position(0, -1),
    ActionType.MOVE_DOWN: Position(0, 1),
    ActionType.MOVE_LEFT: Position(-1, 0),
    ActionType.MOVE_RIGHT: Position(1, 0),
}


class Agent:
    """Agent class representing a special forces unit"""

    def __init__(self, agent_id: str, initial_pos: Position):
        self.id = agent_id
        self.pos = initial_pos
        self.hp = 50
        self.max_hp = 50
        self.damage = 10
        self.range = 2
        self.attack_cooldown = 0
        self.max_attack_cooldown = 2
        self.is_alive = True
        self.is_guarding = False  # Whether guarding VIP

    def take_damage(self, damage: int) -> bool:
        """Apply damage to agent"""
        self.hp = max(0, self.hp - damage)
        self.is_alive = self.hp > 0
        return self.is_alive

    def can_attack(self) -> bool:
        """Check if agent can attack"""
        return self.is_alive and self.attack_cooldown == 0

    def attack(self) -> None:
        """Set attack cooldown after attacking"""
        self.attack_cooldown = self.max_attack_cooldown

    def update_cooldown(self) -> None:
        """Update attack cooldown"""
        if self.attack_cooldown > 0:
            self.attack_cooldown -= 1

    def get_state(self) -> Dict[str, Any]:
        """Get agent state for observation"""
        return {
            'pos': self.pos,
            'hp': self.hp,
            'max_hp': self.max_hp,
            'damage': self.damage,
            'range': self.range,
            'attack_cooldown': self.attack_cooldown,
            'is_alive': self.is_alive,
            'is_guarding': self.is_guarding
        }


class VIP:
    """VIP class representing the scientist to be escorted"""

    def __init__(self, initial_pos: Position, target_pos: Position):
        self.pos = initial_pos
        self.target_pos = target_pos
        self.hp = 60
        self.max_hp = 60
        self.is_alive = True
        self.vision_range = 2
        self.move_cooldown = 0
        self.max_move_cooldown = 2  # Moves every 2 steps
        self.is_under_attack = False

    def take_damage(self, damage: int, damage_reduction: float = 1.0) -> bool:
        """Apply damage to VIP"""
        actual_damage = int(damage * damage_reduction)
        self.hp = max(0, self.hp - actual_damage)
        self.is_alive = self.hp > 0
        return self.is_alive

    def can_move(self) -> bool:
        """Check if VIP can move"""
        return self.is_alive and self.move_cooldown == 0

    def move(self) -> None:
        """Set move cooldown after moving"""
        self.move_cooldown = self.max_move_cooldown

    def update_cooldown(self) -> None:
        """Update move cooldown"""
        if self.move_cooldown > 0:
            self.move_cooldown -= 1

    def get_state(self) -> Dict[str, Any]:
        """Get VIP state for observation"""
        return {
            'pos': self.pos,
            'target_pos': self.target_pos,
            'hp': self.hp,
            'max_hp': self.max_hp,
            'is_alive': self.is_alive,
            'vision_range': self.vision_range,
            'move_cooldown': self.move_cooldown,
            'is_under_attack': self.is_under_attack
        }


class Threat:
    """Threat class representing enemy units"""

    def __init__(self, threat_id: str, threat_type: ThreatType, pos: Position):
        self.id = threat_id
        self.type = threat_type
        self.pos = pos
        self.is_alive = True
        self.attack_cooldown = 0

        # Set attributes based on threat type
        if threat_type == ThreatType.RUSHER:
            self.hp = 40
            self.max_hp = 40
            self.damage = 8
            self.range = 1
            self.move_range = 1
            self.max_attack_cooldown = 1
        else:  # SHOOTER
            self.hp = 30
            self.max_hp = 30
            self.damage = 15
            self.range = 5
            self.move_range = 0  # Fixed position
            self.max_attack_cooldown = 3

    def take_damage(self, damage: int) -> bool:
        """Apply damage to threat"""
        self.hp = max(0, self.hp - damage)
        self.is_alive = self.hp > 0
        return self.is_alive

    def can_attack(self) -> bool:
        """Check if threat can attack"""
        return self.is_alive and self.attack_cooldown == 0

    def attack(self) -> None:
        """Set attack cooldown after attacking"""
        self.attack_cooldown = self.max_attack_cooldown

    def update_cooldown(self) -> None:
        """Update attack cooldown"""
        if self.attack_cooldown > 0:
            self.attack_cooldown -= 1

    def get_state(self) -> Dict[str, Any]:
        """Get threat state for observation"""
        return {
            'id': self.id,
            'type': self.type,
            'pos': self.pos,
            'hp': self.hp,
            'max_hp': self.max_hp,
            'damage': self.damage,
            'range': self.range,
            'move_range': self.move_range,
            'attack_cooldown': self.attack_cooldown,
            'is_alive': self.is_alive
        }


class GameState:
    """Game state class managing the overall environment state"""

    def __init__(self, grid_size: int = 12):
        self.grid_size = grid_size
        self.current_step = 0
        self.max_steps = 200
        self.is_terminated = False
        self.termination_reason = None
        self.total_reward = 0.0

        # Initialize terrain
        self.terrain = self._generate_terrain()

        # Initialize VIP
        self.vip = VIP(
            initial_pos=Position(1, 1),
            target_pos=Position(10, 10)
        )

        # Initialize agents
        self.agents = {}
        self._initialize_agents()

        # Initialize threats
        self.threats = {}
        self.threat_spawn_timer = 0
        self.threat_spawn_interval = 8  # Initial interval

        # Communication system
        self.messages = []  # List of recent messages
        self.max_messages = 10

        # Statistics
        self.stats = {
            'vip_damage_taken': 0,
            'threats_killed': 0,
            'agents_killed': 0,
            'messages_sent': 0,
            'body_blocks': 0,  # VIP damage prevented by guards
            'long_range_kills': 0,  # Threats killed from >=6 units away
            'vip_distance_to_target': 0,
            'agents_adjacent_to_vip': 0,
            'agents_ahead_of_vip': 0,
            'agent_spread': 0,
        }

    def _generate_terrain(self) -> np.ndarray:
        """Generate terrain map"""
        terrain = np.full((self.grid_size, self.grid_size), TerrainType.OPEN, dtype=object)

        # Add rivers (impassable)
        # Add some vertical and horizontal rivers
        for i in range(3, 9):
            terrain[i, 6] = TerrainType.RIVER
        for j in range(3, 9):
            terrain[6, j] = TerrainType.RIVER

        # Add forests (damage reduction)
        forest_positions = [
            (2, 2), (2, 3), (3, 2), (3, 3),  # Forest near start
            (8, 8), (8, 9), (9, 8), (9, 9),  # Forest near target
            (5, 2), (5, 3), (6, 8), (7, 8),  # Additional forests
        ]

        for x, y in forest_positions:
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                terrain[x, y] = TerrainType.FOREST

        return terrain

    def _initialize_agents(self) -> None:
        """Initialize agents around VIP"""
        # Start positions around VIP (1,1)
        start_positions = [
            Position(0, 1), Position(1, 0), Position(2, 1),
            Position(1, 2), Position(0, 0), Position(2, 0)
        ]

        # Filter valid positions (within bounds and not blocked)
        valid_positions = []
        for pos in start_positions:
            if (0 <= pos.x < self.grid_size and 0 <= pos.y < self.grid_size and
                self.terrain[pos.x, pos.y] != TerrainType.RIVER):
                valid_positions.append(pos)

        # Create agents (3 agents)
        num_agents = min(3, len(valid_positions))
        for i in range(num_agents):
            agent_id = f"agent_{i}"
            pos = valid_positions[i]
            self.agents[agent_id] = Agent(agent_id, pos)

    def spawn_threat(self) -> None:
        """Spawn a new threat"""
        if len(self.threats) >= 5:  # Max threats
            return

        # Determine spawn location (ahead of VIP in target direction)
        vip_to_target = self.vip.target_pos - self.vip.pos
        spawn_center = self.vip.pos + Position(
            int(vip_to_target.x * 0.7),  # 70% towards target
            int(vip_to_target.y * 0.7)
        )

        # Find valid spawn positions within 6-9 grid units of VIP
        spawn_positions = []
        for dx in range(-9, 10):
            for dy in range(-9, 10):
                x = spawn_center.x + dx
                y = spawn_center.y + dy
                distance = abs(dx) + abs(dy)  # Manhattan distance
                if (6 <= distance <= 9 and
                    0 <= x < self.grid_size and 0 <= y < self.grid_size and
                    self.terrain[x, y] != TerrainType.RIVER):
                    spawn_positions.append(Position(x, y))

        if spawn_positions:
            spawn_pos = np.random.choice(len(spawn_positions))
            spawn_pos = spawn_positions[spawn_pos]

            # Choose threat type (60% Rusher, 40% Shooter)
            threat_type = ThreatType.RUSHER if np.random.random() < 0.6 else ThreatType.SHOOTER
            threat_id = f"threat_{len(self.threats)}"

            threat = Threat(threat_id, threat_type, spawn_pos)
            self.threats[threat_id] = threat

    def update_threat_spawn_interval(self) -> None:
        """Update threat spawn interval based on VIP HP"""
        if self.vip.hp > 40:
            self.threat_spawn_interval = 6  # More threats when VIP is healthy
        elif self.vip.hp < 20:
            self.threat_spawn_interval = 12  # Fewer threats when VIP is critical
        else:
            self.threat_spawn_interval = 8  # Normal interval

    def add_message(self, agent_id: str, message_type: MessageType) -> None:
        """Add a message to the communication system"""
        message = {
            'sender': agent_id,
            'type': message_type,
            'step': self.current_step
        }
        self.messages.append(message)

        # Keep only recent messages
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

        self.stats['messages_sent'] += 1

    def is_valid_position(self, pos: Position) -> bool:
        """Check if position is valid (within bounds and not blocked)"""
        return (0 <= pos.x < self.grid_size and
                0 <= pos.y < self.grid_size and
                self.terrain[pos.x, pos.y] != TerrainType.RIVER)

    def is_position_occupied(self, pos: Position, exclude_id: str = None) -> bool:
        """Check if position is occupied by another entity"""
        if pos == self.vip.pos and exclude_id != "vip":
            return True

        for agent_id, agent in self.agents.items():
            if agent_id != exclude_id and agent.is_alive and agent.pos == pos:
                return True

        for threat_id, threat in self.threats.items():
            if threat_id != exclude_id and threat.is_alive and threat.pos == pos:
                return True

        return False

    def calculate_agent_spread(self) -> float:
        """Calculate average distance between agents"""
        alive_agents = [agent for agent in self.agents.values() if agent.is_alive]
        if len(alive_agents) < 2:
            return 0.0

        total_distance = 0.0
        count = 0
        for i, agent1 in enumerate(alive_agents):
            for agent2 in alive_agents[i+1:]:
                total_distance += agent1.pos.manhattan_distance(agent2.pos)
                count += 1

        return total_distance / count if count > 0 else 0.0

    def update_statistics(self) -> None:
        """Update game statistics"""
        self.stats['vip_distance_to_target'] = self.vip.pos.manhattan_distance(self.vip.target_pos)

        # Count agents adjacent to VIP
        adjacent_count = 0
        for agent in self.agents.values():
            if agent.is_alive and agent.pos.manhattan_distance(self.vip.pos) == 1:
                adjacent_count += 1
        self.stats['agents_adjacent_to_vip'] = adjacent_count

        # Count agents ahead of VIP (towards target direction)
        ahead_count = 0
        vip_to_target = self.vip.target_pos - self.vip.pos
        for agent in self.agents.values():
            if agent.is_alive:
                agent_to_vip = agent.pos - self.vip.pos
                # Check if agent is in the general direction towards target
                if (agent_to_vip.x * vip_to_target.x >= 0 and
                    agent_to_vip.y * vip_to_target.y >= 0 and
                    agent.pos.manhattan_distance(self.vip.pos) <= 3):
                    ahead_count += 1
        self.stats['agents_ahead_of_vip'] = ahead_count

        # Calculate agent spread
        self.stats['agent_spread'] = self.calculate_agent_spread()

    def get_observation_for_agent(self, agent_id: str) -> Dict[str, Any]:
        """Get observation for a specific agent"""
        if agent_id not in self.agents:
            return {}

        agent = self.agents[agent_id]
        if not agent.is_alive:
            return {}

        vision_range = 4  # Fixed vision range for all agents

        observation = {
            'self': self._get_agent_self_observation(agent),
            'vip': self._get_vip_observation(agent, vision_range),
            'teammates': self._get_teammates_observation(agent, vision_range),
            'threats': self._get_threats_observation(agent, vision_range),
            'communication': self._get_communication_observation(),
            'terrain': self._get_terrain_observation(agent, vision_range),
            'step': self.current_step,
            'max_steps': self.max_steps
        }

        return observation

    def _get_agent_self_observation(self, agent: Agent) -> Dict[str, Any]:
        """Get self observation for agent"""
        vip_dist = agent.pos.manhattan_distance(self.vip.pos)
        target_dist = agent.pos.manhattan_distance(self.vip.target_pos)

        return {
            'pos': agent.pos,
            'hp': agent.hp / agent.max_hp,  # Normalized
            'attack_cooldown': agent.attack_cooldown / agent.max_attack_cooldown,
            'is_guarding': int(agent.is_guarding),
            'vip_distance': vip_dist / 20.0,  # Normalized
            'target_distance': target_dist / 20.0,  # Normalized
            'is_in_forest': int(self.terrain[agent.pos.x, agent.pos.y] == TerrainType.FOREST)
        }

    def _get_vip_observation(self, agent: Agent, vision_range: int) -> Dict[str, Any]:
        """Get VIP observation for agent"""
        dist = agent.pos.manhattan_distance(self.vip.pos)
        in_range = dist <= vision_range

        obs = {
            'visible': int(in_range),
            'hp': 0.0,
            'relative_pos': (0, 0),
            'is_under_attack': 0,
            'is_adjacent': 0
        }

        if in_range:
            obs['hp'] = self.vip.hp / self.vip.max_hp
            obs['relative_pos'] = (self.vip.pos.x - agent.pos.x, self.vip.pos.y - agent.pos.y)
            obs['is_under_attack'] = int(self.vip.is_under_attack)
            obs['is_adjacent'] = int(dist == 1)

        return obs

    def _get_teammates_observation(self, agent: Agent, vision_range: int) -> List[Dict[str, Any]]:
        """Get teammates observation for agent"""
        teammates_obs = []
        agent_id = agent.id  # Fix: get agent_id from agent

        for teammate_id, teammate in self.agents.items():
            if teammate_id == agent_id or not teammate.is_alive:
                continue

            dist = agent.pos.manhattan_distance(teammate.pos)
            in_range = dist <= vision_range

            if in_range:
                teammates_obs.append({
                    'id': teammate_id,
                    'relative_pos': (teammate.pos.x - agent.pos.x, teammate.pos.y - agent.pos.y),
                    'hp': teammate.hp / teammate.max_hp,
                    'is_adjacent_to_vip': int(teammate.pos.manhattan_distance(self.vip.pos) == 1),
                    'is_guarding': int(teammate.is_guarding),
                    'attack_cooldown': teammate.attack_cooldown / teammate.max_attack_cooldown
                })

        return teammates_obs

    def _get_threats_observation(self, agent: Agent, vision_range: int) -> List[Dict[str, Any]]:
        """Get threats observation for agent"""
        threats_obs = []

        for threat in self.threats.values():
            if not threat.is_alive:
                continue

            dist = agent.pos.manhattan_distance(threat.pos)
            in_range = dist <= vision_range

            if in_range:
                threats_obs.append({
                    'type': threat.type.value,
                    'relative_pos': (threat.pos.x - agent.pos.x, threat.pos.y - agent.pos.y),
                    'hp': threat.hp / threat.max_hp,
                    'attack_cooldown': threat.attack_cooldown / threat.max_attack_cooldown
                })

        return threats_obs

    def _get_communication_observation(self) -> List[Dict[str, Any]]:
        """Get recent communication observations"""
        # Return last 3 messages
        recent_messages = self.messages[-3:] if self.messages else []
        comm_obs = []

        for msg in recent_messages:
            comm_obs.append({
                'type': msg['type'],
                'age': self.current_step - msg['step']
            })

        return comm_obs

    def _get_terrain_observation(self, agent: Agent, vision_range: int) -> Dict[str, Any]:
        """Get terrain observation for agent"""
        terrain_obs = {}

        for dx in range(-vision_range, vision_range + 1):
            for dy in range(-vision_range, vision_range + 1):
                x = agent.pos.x + dx
                y = agent.pos.y + dy
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    terrain_type = self.terrain[x, y]
                    if terrain_type != TerrainType.OPEN:
                        terrain_obs[(dx, dy)] = terrain_type.value

        return terrain_obs

    def get_global_state(self) -> np.ndarray:
        """Get global state representation for centralized training"""
        state_features = []

        # VIP state
        state_features.extend([
            self.vip.pos.x / self.grid_size,
            self.vip.pos.y / self.grid_size,
            self.vip.hp / self.vip.max_hp,
            self.vip.is_under_attack
        ])

        # Agents state
        for agent in self.agents.values():
            if agent.is_alive:
                state_features.extend([
                    agent.pos.x / self.grid_size,
                    agent.pos.y / self.grid_size,
                    agent.hp / agent.max_hp,
                    agent.is_guarding
                ])
            else:
                state_features.extend([0.0, 0.0, 0.0, 0.0])

        # Threats state (up to 5 threats)
        for i in range(5):
            if i < len(self.threats):
                threat = list(self.threats.values())[i]
                if threat.is_alive:
                    state_features.extend([
                        threat.pos.x / self.grid_size,
                        threat.pos.y / self.grid_size,
                        threat.hp / threat.max_hp,
                        int(threat.type == ThreatType.RUSHER)
                    ])
                else:
                    state_features.extend([0.0, 0.0, 0.0, 0.0])
            else:
                state_features.extend([0.0, 0.0, 0.0, 0.0])

        # Statistics
        state_features.extend([
            self.stats['vip_distance_to_target'] / 20.0,
            self.stats['agents_adjacent_to_vip'] / 3.0,
            self.stats['agents_ahead_of_vip'] / 3.0,
            self.stats['agent_spread'] / 10.0,
            self.current_step / self.max_steps
        ])

        return np.array(state_features, dtype=np.float32)