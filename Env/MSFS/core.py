"""
MSFS Environment Core Components

This module contains the core classes and utilities for the MSFS environment,
including agent types, workstations, orders, and game state management.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from enum import IntEnum
from dataclasses import dataclass, field
import random


class OrderType(IntEnum):
    """Order type enumeration"""
    SIMPLE = 0  # S-type order
    COMPLEX = 1  # C-type order


class WorkstationType(IntEnum):
    """Workstation type enumeration"""
    RAW = 0      # Raw material preparation
    ASSEMBLY = 1 # Assembly station
    PACKING = 2  # Packing station


class ActionType(IntEnum):
    """Action type enumeration"""
    WAIT = 0
    MOVE_TO_RAW = 1
    MOVE_TO_ASSEMBLY = 2
    MOVE_TO_PACKING = 3
    PULL_ORDER = 4
    START_PROCESSING = 5
    COMPLETE_STAGE = 6
    DELIVER_ORDER = 7


@dataclass
class Order:
    """Represents an order in the manufacturing pipeline"""
    order_id: str
    order_type: OrderType
    current_stage: int = 0  # 0: not pulled, 1: Raw, 2: Assembly, 3: Packing, 4: delivered
    processing_progress: int = 0  # Progress within current stage
    position: Optional[WorkstationType] = None  # Current workstation position

    def get_processing_time(self, stage: WorkstationType) -> int:
        """Get processing time for a specific stage"""
        if self.order_type == OrderType.SIMPLE:
            times = {WorkstationType.RAW: 1, WorkstationType.ASSEMBLY: 2, WorkstationType.PACKING: 1}
        else:  # COMPLEX
            times = {WorkstationType.RAW: 2, WorkstationType.ASSEMBLY: 3, WorkstationType.PACKING: 1}
        return times.get(stage, 1)

    def get_value(self) -> float:
        """Get order value for reward calculation"""
        return 5.0 if self.order_type == OrderType.SIMPLE else 10.0

    def is_complete(self) -> bool:
        """Check if order is completely processed"""
        return self.current_stage == 4  # Delivered


@dataclass
class Workstation:
    """Represents a workstation in the pipeline"""
    workstation_type: WorkstationType
    queue: List[Order] = field(default_factory=list)
    current_order: Optional[Order] = None
    capacity: int = 1

    def add_order(self, order: Order) -> bool:
        """Add order to workstation queue"""
        if len(self.queue) < 20:  # Max queue size
            self.queue.append(order)
            order.position = self.workstation_type
            return True
        return False

    def get_next_order(self) -> Optional[Order]:
        """Get next order from queue"""
        if self.queue and not self.current_order:
            self.current_order = self.queue.pop(0)
            return self.current_order
        return None

    def complete_current_order(self) -> Optional[Order]:
        """Complete processing of current order"""
        if self.current_order:
            order = self.current_order
            self.current_order = None
            return order
        return None

    def get_queue_length(self) -> int:
        """Get total queue length including current order"""
        return len(self.queue) + (1 if self.current_order else 0)


@dataclass
class Agent:
    """Represents an agent (robot) in the environment"""
    agent_id: str
    current_workstation: WorkstationType
    move_cooldown: int = 0
    carrying_order: Optional[Order] = None
    specialization_count: Dict[WorkstationType, int] = field(default_factory=lambda: {WorkstationType.RAW: 0, WorkstationType.ASSEMBLY: 0, WorkstationType.PACKING: 0})
    consecutive_specialization: Dict[WorkstationType, int] = field(default_factory=lambda: {WorkstationType.RAW: 0, WorkstationType.ASSEMBLY: 0, WorkstationType.PACKING: 0})

    def can_move(self) -> bool:
        """Check if agent can move"""
        return self.move_cooldown == 0

    def move_to(self, workstation: WorkstationType) -> None:
        """Move agent to workstation"""
        self.current_workstation = workstation
        self.move_cooldown = 1

    def update_cooldown(self) -> None:
        """Update move cooldown"""
        if self.move_cooldown > 0:
            self.move_cooldown -= 1

    def is_busy(self) -> bool:
        """Check if agent is busy processing"""
        return self.move_cooldown > 0


@dataclass
class GameState:
    """Represents the current game state"""
    current_step: int = 0
    max_steps: int = 50

    # Workstations
    workstations: Dict[WorkstationType, Workstation] = field(default_factory=lambda: {
        WorkstationType.RAW: Workstation(WorkstationType.RAW),
        WorkstationType.ASSEMBLY: Workstation(WorkstationType.ASSEMBLY),
        WorkstationType.PACKING: Workstation(WorkstationType.PACKING)
    })

    # Agents
    agents: Dict[str, Agent] = field(default_factory=dict)

    # Orders
    orders: List[Order] = field(default_factory=list)
    pending_orders: List[Order] = field(default_factory=list)
    completed_orders: List[Order] = field(default_factory=list)
    total_orders_generated: int = 0

    # Statistics
    total_reward: float = 0.0
    orders_completed: int = 0
    simple_orders_completed: int = 0
    complex_orders_completed: int = 0
    efficiency_penalty: float = 0.0

    # Role emergence statistics
    specialization_events: int = 0
    role_switch_events: int = 0
    station_utilization: Dict[WorkstationType, float] = field(default_factory=lambda: {
        WorkstationType.RAW: 0.0,
        WorkstationType.ASSEMBLY: 0.0,
        WorkstationType.PACKING: 0.0
    })

    def is_terminated(self) -> bool:
        """Check if the episode is terminated"""
        return self.current_step >= self.max_steps

    def get_arrival_probability(self) -> float:
        """Get order arrival probability based on current step"""
        if self.current_step < 15:
            return 0.5  # Phase 1: Warm-up
        elif self.current_step < 35:
            return 0.8  # Phase 2: Peak
        else:
            return 0.3  # Phase 3: Wind-down

    def get_order_type_distribution(self) -> float:
        """Get probability of complex order based on current step"""
        if self.current_step < 15:
            return 0.3  # Phase 1: 30% complex
        elif self.current_step < 35:
            return 0.6  # Phase 2: 60% complex
        else:
            return 0.4  # Phase 3: 40% complex

    def generate_order(self) -> Optional[Order]:
        """Generate a new order based on current step"""
        if random.random() < self.get_arrival_probability():
            order_type = OrderType.COMPLEX if random.random() < self.get_order_type_distribution() else OrderType.SIMPLE
            order = Order(
                order_id=f"order_{self.total_orders_generated}",
                order_type=order_type
            )
            self.total_orders_generated += 1
            return order
        return None

    def get_role_emergence_stats(self) -> Dict[str, float]:
        """Get role emergence statistics"""
        stats = {
            'specialization_events': self.specialization_events,
            'role_switch_events': self.role_switch_events,
            'raw_utilization': self.station_utilization[WorkstationType.RAW],
            'assembly_utilization': self.station_utilization[WorkstationType.ASSEMBLY],
            'packing_utilization': self.station_utilization[WorkstationType.PACKING],
        }

        # Calculate agent specialization metrics
        if self.agents:
            for agent_id, agent in self.agents.items():
                for station_type in WorkstationType:
                    stats[f'agent_{agent_id}_{station_type.name.lower()}_specialization'] = agent.specialization_count[station_type]

        return stats

    def get_queue_stats(self) -> Dict[str, int]:
        """Get queue statistics for all workstations"""
        stats = {}
        for station_type in WorkstationType:
            workstation = self.workstations[station_type]
            # Count order types in queue
            simple_count = sum(1 for order in workstation.queue if order.order_type == OrderType.SIMPLE)
            complex_count = sum(1 for order in workstation.queue if order.order_type == OrderType.COMPLEX)

            stats[f'{station_type.name.lower()}_queue_length'] = workstation.get_queue_length()
            stats[f'{station_type.name.lower()}_simple_orders'] = simple_count
            stats[f'{station_type.name.lower()}_complex_orders'] = complex_count

        return stats


class Position:
    """Simple position class for compatibility"""
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __hash__(self):
        return hash((self.x, self.y))

    def __str__(self):
        return f"({self.x}, {self.y})"

    def manhattan_distance(self, other: 'Position') -> int:
        """Calculate Manhattan distance"""
        return abs(self.x - other.x) + abs(self.y - other.y)