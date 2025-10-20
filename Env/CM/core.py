"""
CM (Collaborative Moving) Environment Core Classes

This module defines the core classes for the collaborative box pushing environment.
Agents need to work together to push a box to a target location.
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum


class ActionType(Enum):
    """Action types for agents in the CM environment."""
    STAY = 0
    MOVE_UP = 1
    MOVE_DOWN = 2
    MOVE_LEFT = 3
    MOVE_RIGHT = 4


@dataclass
class Position:
    """Represents a position in the grid."""
    x: int
    y: int

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        if isinstance(other, Position):
            return self.x == other.x and self.y == other.y
        return False

    def distance_to(self, other: 'Position') -> float:
        """Calculate Manhattan distance to another position."""
        return abs(self.x - other.x) + abs(self.y - other.y)

    def is_adjacent(self, other: 'Position') -> bool:
        """Check if this position is adjacent to another position."""
        return self.distance_to(other) == 1

    def copy(self) -> 'Position':
        """Create a copy of this position."""
        return Position(self.x, self.y)


class Box:
    """Represents the box that agents need to push."""

    def __init__(self, position: Position, size: int = 2):
        self.position = position  # Top-left corner
        self.size = size
        self.push_cooldown = 0  # Cooldown after being pushed

    def get_occupied_positions(self) -> List[Position]:
        """Get all positions occupied by the box."""
        positions = []
        for dx in range(self.size):
            for dy in range(self.size):
                positions.append(Position(self.position.x + dx, self.position.y + dy))
        return positions

    def get_center(self) -> Tuple[float, float]:
        """Get the center position of the box."""
        center_x = self.position.x + (self.size - 1) / 2
        center_y = self.position.y + (self.size - 1) / 2
        return center_x, center_y

    def is_position_on_side(self, pos: Position, side: str) -> bool:
        """Check if a position is on a specific side of the box."""
        box_positions = self.get_occupied_positions()

        if side == "top":
            return (pos.x == self.position.x - 1 and
                    self.position.y <= pos.y < self.position.y + self.size)
        elif side == "bottom":
            return (pos.x == self.position.x + self.size and
                    self.position.y <= pos.y < self.position.y + self.size)
        elif side == "left":
            return (pos.y == self.position.y - 1 and
                    self.position.x <= pos.x < self.position.x + self.size)
        elif side == "right":
            return (pos.y == self.position.y + self.size and
                    self.position.x <= pos.x < self.position.x + self.size)

        return False

    def can_move(self, direction: np.ndarray, grid_size: int) -> bool:
        """Check if the box can move in the given direction."""
        new_pos_x = self.position.x + direction[0]
        new_pos_y = self.position.y + direction[1]

        # Check if new position is within bounds
        return (0 <= new_pos_x and 0 <= new_pos_y and
                new_pos_x + self.size <= grid_size and
                new_pos_y + self.size <= grid_size)

    def move(self, direction: np.ndarray):
        """Move the box in the given direction."""
        self.position.x += int(direction[0])
        self.position.y += int(direction[1])
        self.push_cooldown = 1  # Set cooldown after moving


class Goal:
    """Represents the target location for the box."""

    def __init__(self, position: Position, size: int = 2):
        self.position = position  # Top-left corner
        self.size = size

    def get_occupied_positions(self) -> List[Position]:
        """Get all positions occupied by the goal."""
        positions = []
        for dx in range(self.size):
            for dy in range(self.size):
                positions.append(Position(self.position.x + dx, self.position.y + dy))
        return positions

    def get_center(self) -> Tuple[float, float]:
        """Get the center position of the goal."""
        center_x = self.position.x + (self.size - 1) / 2
        center_y = self.position.y + (self.size - 1) / 2
        return center_x, center_y

    def is_achieved(self, box: Box) -> bool:
        """Check if the box has reached the goal."""
        return self.position.x == box.position.x and self.position.y == box.position.y


class Agent:
    """Represents an agent in the CM environment."""

    def __init__(self, agent_id: str, position: Position):
        self.id = agent_id
        self.position = position
        self.last_action = ActionType.STAY
        self.collided = False

    def move(self, action: ActionType, grid_size: int) -> bool:
        """
        Move the agent based on the action.

        Returns:
            bool: True if move was successful, False if hit boundary
        """
        self.last_action = action
        old_pos = self.position.copy()

        if action == ActionType.MOVE_UP:
            self.position.x -= 1
        elif action == ActionType.MOVE_DOWN:
            self.position.x += 1
        elif action == ActionType.MOVE_LEFT:
            self.position.y -= 1
        elif action == ActionType.MOVE_RIGHT:
            self.position.y += 1
        # STAY action keeps position unchanged

        # Check boundary collision
        if (self.position.x < 0 or self.position.x >= grid_size or
            self.position.y < 0 or self.position.y >= grid_size):
            self.position = old_pos  # Revert move
            return False

        return True

    def get_valid_actions(self, grid_size: int) -> List[int]:
        """Get list of valid actions from current position."""
        valid_actions = [ActionType.STAY.value]

        # Check each direction
        if self.position.x > 0:
            valid_actions.append(ActionType.MOVE_UP.value)
        if self.position.x < grid_size - 1:
            valid_actions.append(ActionType.MOVE_DOWN.value)
        if self.position.y > 0:
            valid_actions.append(ActionType.MOVE_LEFT.value)
        if self.position.y < grid_size - 1:
            valid_actions.append(ActionType.MOVE_RIGHT.value)

        return valid_actions

    def is_pushing_box(self, box: Box) -> bool:
        """Check if agent is in position to push the box."""
        return (box.is_position_on_side(self.position, "top") or
                box.is_position_on_side(self.position, "bottom") or
                box.is_position_on_side(self.position, "left") or
                box.is_position_on_side(self.position, "right"))

    def get_push_side(self, box: Box) -> Optional[str]:
        """Get which side of the box the agent is on, if any."""
        if box.is_position_on_side(self.position, "top"):
            return "top"
        elif box.is_position_on_side(self.position, "bottom"):
            return "bottom"
        elif box.is_position_on_side(self.position, "left"):
            return "left"
        elif box.is_position_on_side(self.position, "right"):
            return "right"
        return None


class CMGameState:
    """Manages the overall game state."""

    def __init__(self, agents: List[Agent], box: Box, goal: Goal, grid_size: int):
        self.agents = {agent.id: agent for agent in agents}
        self.box = box
        self.goal = goal
        self.grid_size = grid_size
        self.current_step = 0
        self.total_score = 0
        self.history = []  # Record of actions and outcomes

    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get an agent by ID."""
        return self.agents.get(agent_id)

    def get_occupied_positions(self) -> List[Position]:
        """Get all occupied positions in the grid."""
        occupied = []

        # Add box positions
        occupied.extend(self.box.get_occupied_positions())

        # Add agent positions
        for agent in self.agents.values():
            occupied.append(agent.position)

        return occupied

    def is_position_occupied(self, pos: Position, exclude_agent: Optional[str] = None) -> bool:
        """Check if a position is occupied."""
        # Check box positions
        for box_pos in self.box.get_occupied_positions():
            if pos == box_pos:
                return True

        # Check agent positions
        for agent_id, agent in self.agents.items():
            if agent_id != exclude_agent and pos == agent.position:
                return True

        return False

    def is_complete(self) -> bool:
        """Check if the current episode is complete."""
        return self.goal.is_achieved(self.box)

    def copy(self) -> 'CMGameState':
        """Create a deep copy of the game state."""
        # Create new agents
        new_agents = []
        for agent in self.agents.values():
            new_agent = Agent(agent.id, agent.position.copy())
            new_agent.last_action = agent.last_action
            new_agent.collided = agent.collided
            new_agents.append(new_agent)

        # Create new box and goal
        new_box = Box(self.box.position.copy(), self.box.size)
        new_box.push_cooldown = self.box.push_cooldown

        new_goal = Goal(self.goal.position.copy(), self.goal.size)

        # Create new state
        new_state = CMGameState(new_agents, new_box, new_goal, self.grid_size)
        new_state.current_step = self.current_step
        new_state.total_score = self.total_score
        new_state.history = self.history.copy()

        return new_state

    def get_pushing_agents(self) -> List[str]:
        """Get list of agent IDs that are in pushing position."""
        pushing_agents = []
        for agent_id, agent in self.agents.items():
            if agent.is_pushing_box(self.box):
                pushing_agents.append(agent_id)
        return pushing_agents

    def get_push_sides(self) -> List[str]:
        """Get unique sides that have agents pushing."""
        sides = set()
        for agent in self.agents.values():
            side = agent.get_push_side(self.box)
            if side:
                sides.add(side)
        return list(sides)

    def calculate_push_direction(self, sides: List[str]) -> np.ndarray:
        """Calculate the push direction based on active pushing sides."""
        direction = np.array([0, 0])

        # Vertical push
        if "top" in sides and "bottom" not in sides:
            direction[0] = -1  # Push up
        elif "bottom" in sides and "top" not in sides:
            direction[0] = 1   # Push down

        # Horizontal push
        if "left" in sides and "right" not in sides:
            direction[1] = -1  # Push left
        elif "right" in sides and "left" not in sides:
            direction[1] = 1   # Push right

        return direction