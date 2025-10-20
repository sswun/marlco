"""
HRG Environment Core Components

This module contains the core classes and utilities for the HRG environment,
including agent types, resources, and game state management.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from enum import IntEnum
from dataclasses import dataclass, field
import random


class AgentType(IntEnum):
    """Agent type enumeration"""
    SCOUT = 0
    WORKER = 1
    TRANSPORTER = 2


class ResourceType(IntEnum):
    """Resource type enumeration"""
    GOLD = 0
    WOOD = 1


class ActionType(IntEnum):
    """Action type enumeration"""
    MOVE_NORTH = 0
    MOVE_SOUTH = 1
    MOVE_WEST = 2
    MOVE_EAST = 3
    GATHER = 4
    TRANSFER = 5
    DEPOSIT = 6
    WAIT = 7


@dataclass
class AgentConfig:
    """Agent configuration"""
    agent_type: AgentType
    vision_range: int
    move_speed: float
    carry_capacity: int
    gather_time: int
    energy_consumption_move: float
    energy_consumption_gather: float
    energy_consumption_transfer: float
    initial_position: Tuple[int, int]


@dataclass
class ResourceConfig:
    """Resource configuration"""
    resource_type: ResourceType
    value: float
    quantity_per_unit: int
    gather_difficulty: int
    respawn_time: int


@dataclass
class Position:
    """Position on the grid"""
    x: int
    y: int

    def __hash__(self):
        return hash((self.x, self.y))

    def distance_to(self, other: 'Position') -> int:
        """Calculate Manhattan distance to another position"""
        return abs(self.x - other.x) + abs(self.y - other.y)

    def is_adjacent(self, other: 'Position') -> bool:
        """Check if this position is adjacent to another"""
        return self.distance_to(other) == 1


@dataclass
class Resource:
    """Resource on the grid"""
    position: Position
    resource_type: ResourceType
    remaining_quantity: int
    respawn_timer: int = 0
    is_active: bool = True

    def gather(self, amount: int = 1) -> int:
        """Gather resource, returns amount actually gathered"""
        if not self.is_active or self.remaining_quantity <= 0:
            return 0

        gathered = min(amount, self.remaining_quantity)
        self.remaining_quantity -= gathered

        if self.remaining_quantity <= 0:
            self.is_active = False
            self.respawn_timer = self._get_respawn_time()

        return gathered

    def _get_respawn_time(self) -> int:
        """Get respawn time based on resource type"""
        return 50 if self.resource_type == ResourceType.GOLD else 50

    def update_respawn(self):
        """Update respawn timer - optimized version"""
        if not self.is_active and self.respawn_timer > 0:
            self.respawn_timer -= 1
            if self.respawn_timer == 0:
                self.respawn()

    def respawn(self):
        """Respawn resource with full quantity"""
        self.is_active = True
        self.remaining_quantity = self._get_initial_quantity()
        self.respawn_timer = 0

    def _get_initial_quantity(self) -> int:
        """Get initial quantity based on resource type"""
        return 5 if self.resource_type == ResourceType.GOLD else 10


class Agent:
    """Agent in the HRG environment"""

    def __init__(self, agent_id: str, config: AgentConfig):
        self.id = agent_id
        self.type = config.agent_type
        self.config = config

        # Position and movement
        self.position = Position(*config.initial_position)
        self.move_points = 0.0

        # Resources and inventory
        self.inventory = {
            ResourceType.GOLD: 0,
            ResourceType.WOOD: 0
        }

        # Energy and cooldowns
        self.energy = 100.0
        self.action_cooldown = 0

        # Gathering state
        self.gathering_target = None
        self.gathering_progress = 0

    @property
    def is_at_base(self) -> bool:
        """Check if agent is at base position"""
        return self.position.x == 0 and self.position.y == 0

    @property
    def is_carrying_resources(self) -> bool:
        """Check if agent is carrying any resources"""
        return any(count > 0 for count in self.inventory.values())

    @property
    def carry_weight(self) -> int:
        """Get total carry weight"""
        return sum(self.inventory.values())

    @property
    def can_carry_more(self) -> bool:
        """Check if agent can carry more resources"""
        return self.carry_weight < self.config.carry_capacity

    @property
    def total_inventory_value(self) -> float:
        """Calculate total value of inventory"""
        resource_values = {
            ResourceType.GOLD: 10.0,
            ResourceType.WOOD: 2.0
        }
        return sum(self.inventory[rtype] * resource_values[rtype]
                  for rtype in ResourceType)

    def reset_turn(self):
        """Reset agent turn state"""
        self.move_points = self.config.move_speed
        if self.action_cooldown > 0:
            self.action_cooldown -= 1

    def consume_energy(self, amount: float):
        """Consume energy"""
        self.energy = max(0, self.energy - amount)

    def can_perform_action(self, action: ActionType) -> bool:
        """Check if agent can perform given action"""
        if self.action_cooldown > 0:
            return False

        if self.energy <= 0:
            return False

        # Type-specific restrictions
        if action == ActionType.GATHER and self.type == AgentType.SCOUT:
            return False

        if action == ActionType.DEPOSIT and self.type != AgentType.TRANSPORTER:
            return False

        if action == ActionType.GATHER and not self.can_carry_more:
            return False

        if action == ActionType.DEPOSIT and not self.is_at_base:
            return False

        return True

    def add_resources(self, resource_type: ResourceType, amount: int):
        """Add resources to inventory"""
        if self.can_carry_more:
            can_add = min(amount, self.config.carry_capacity - self.carry_weight)
            self.inventory[resource_type] += can_add
            return can_add
        return 0

    def remove_resources(self, resource_type: ResourceType, amount: int) -> int:
        """Remove resources from inventory"""
        available = self.inventory[resource_type]
        removed = min(amount, available)
        self.inventory[resource_type] -= removed
        return removed

    def clear_inventory(self) -> Dict[ResourceType, int]:
        """Clear all inventory and return removed resources"""
        removed = self.inventory.copy()
        for rtype in ResourceType:
            self.inventory[rtype] = 0
        return removed


class GameState:
    """Game state management"""

    def __init__(self, grid_size: int = 10, num_obstacles: int = 10):
        self.grid_size = grid_size
        self.base_position = Position(0, 0)

        # Initialize grid
        self.grid = np.zeros((grid_size, grid_size), dtype=int)
        self.obstacles = set()
        self._generate_obstacles(num_obstacles)

        # Agents and resources
        self.agents: Dict[str, Agent] = {}
        self.resources: List[Resource] = []

        # Game state
        self.current_step = 0
        self.max_steps = 200
        self.total_score = 0
        self.deposited_resources = {
            ResourceType.GOLD: 0,
            ResourceType.WOOD: 0
        }

        # Communication
        self.message_history: List[Dict[str, Any]] = []
        self.max_message_history = 3

    def _generate_obstacles(self, num_obstacles: int):
        """Generate random obstacles on the grid"""
        available_positions = []
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                # Don't place obstacles at base or adjacent to base
                if (x, y) != (0, 0) and abs(x) + abs(y) > 1:
                    available_positions.append(Position(x, y))

        # Randomly select obstacle positions
        obstacle_positions = random.sample(
            available_positions,
            min(num_obstacles, len(available_positions))
        )
        self.obstacles = set(obstacle_positions)

        # Mark obstacles on grid
        for pos in self.obstacles:
            self.grid[pos.y, pos.x] = -1

    def is_valid_position(self, position: Position) -> bool:
        """Check if position is valid (within bounds and not obstacle)"""
        if (position.x < 0 or position.x >= self.grid_size or
            position.y < 0 or position.y >= self.grid_size):
            return False
        return position not in self.obstacles

    def get_visible_positions(self, agent: Agent) -> List[Position]:
        """Get positions visible to agent - optimized version"""
        visible = []
        vision_range = agent.config.vision_range

        # Use simple Manhattan distance for speed
        for x in range(max(0, agent.position.x - vision_range),
                      min(self.grid_size, agent.position.x + vision_range + 1)):
            for y in range(max(0, agent.position.y - vision_range),
                          min(self.grid_size, agent.position.y + vision_range + 1)):
                pos = Position(x, y)
                # Fast distance check
                if (abs(pos.x - agent.position.x) + abs(pos.y - agent.position.y)) <= vision_range:
                    visible.append(pos)

        return visible

    def add_agent(self, agent: Agent):
        """Add agent to game state"""
        self.agents[agent.id] = agent

    def add_resource(self, resource: Resource):
        """Add resource to game state"""
        self.resources.append(resource)

    def get_resource_at(self, position: Position) -> Optional[Resource]:
        """Get resource at specific position"""
        for resource in self.resources:
            if resource.position == position and resource.is_active:
                return resource
        return None

    def get_agents_at(self, position: Position) -> List[Agent]:
        """Get all agents at specific position"""
        return [agent for agent in self.agents.values()
                if agent.position == position]

    def deposit_resources(self, resources: Dict[ResourceType, int]):
        """Deposit resources at base and update score"""
        for rtype, amount in resources.items():
            self.deposited_resources[rtype] += amount
            if rtype == ResourceType.GOLD:
                self.total_score += amount * 10
            elif rtype == ResourceType.WOOD:
                self.total_score += amount * 2

    def add_message(self, agent_id: str, message: np.ndarray, metadata: Dict = None):
        """Add message to history"""
        self.message_history.append({
            'agent_id': agent_id,
            'message': message,
            'step': self.current_step,
            'metadata': metadata or {}
        })

        # Keep only recent messages
        if len(self.message_history) > self.max_message_history:
            self.message_history.pop(0)

    def is_terminal(self) -> bool:
        """Check if game is in terminal state"""
        return self.current_step >= self.max_steps

    def update(self):
        """Update game state for one step - optimized version"""
        self.current_step += 1

        # Only update resources that need updating (inactive ones with timers)
        for resource in self.resources:
            if not resource.is_active and resource.respawn_timer > 0:
                resource.respawn_timer -= 1
                if resource.respawn_timer == 0:
                    resource.respawn()


# Predefined agent configurations
AGENT_CONFIGS = {
    AgentType.SCOUT: AgentConfig(
        agent_type=AgentType.SCOUT,
        vision_range=5,
        move_speed=2.0,
        carry_capacity=0,
        gather_time=0,
        energy_consumption_move=0.05,
        energy_consumption_gather=0.0,
        energy_consumption_transfer=0.0,
        initial_position=(0, 1)
    ),
    AgentType.WORKER: AgentConfig(
        agent_type=AgentType.WORKER,
        vision_range=3,
        move_speed=1.0,
        carry_capacity=2,
        gather_time=2,
        energy_consumption_move=0.02,
        energy_consumption_gather=0.08,
        energy_consumption_transfer=0.0,
        initial_position=(1, 1)
    ),
    AgentType.TRANSPORTER: AgentConfig(
        agent_type=AgentType.TRANSPORTER,
        vision_range=4,
        move_speed=1.5,
        carry_capacity=5,
        gather_time=0,
        energy_consumption_move=0.03,
        energy_consumption_gather=0.0,
        energy_consumption_transfer=0.1,
        initial_position=(0, 0)
    )
}

# Resource configurations
RESOURCE_CONFIGS = {
    ResourceType.GOLD: ResourceConfig(
        resource_type=ResourceType.GOLD,
        value=10.0,
        quantity_per_unit=2,
        gather_difficulty=4,
        respawn_time=50
    ),
    ResourceType.WOOD: ResourceConfig(
        resource_type=ResourceType.WOOD,
        value=2.0,
        quantity_per_unit=1,
        gather_difficulty=2,
        respawn_time=50
    )
}