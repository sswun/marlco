"""
HRG (Heterogeneous Resource Gathering) Environment - Ultra Fast Training Version

Extreme optimization for maximum training speed with minimal agent count.
Key optimizations:
- Minimal agent configuration (2 agents total)
- Simplified resource management
- Ultra-fast observation generation
- Eliminated all unnecessary computations
- Streamlined state management
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from gymnasium import spaces
from dataclasses import replace

from .core import (
    Agent, AgentType, ResourceType, ActionType, Position, Resource,
    AGENT_CONFIGS, RESOURCE_CONFIGS
)

logger = logging.getLogger(__name__)


class UltraFastConfig:
    """Ultra fast training configuration with minimal agents"""

    def __init__(self,
                 grid_size: int = 6,
                 max_steps: int = 80,
                 num_obstacles: int = 2,
                 num_gold: int = 1,
                 num_wood: int = 4,
                 seed: Optional[int] = None):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.num_obstacles = num_obstacles
        self.num_gold = num_gold
        self.num_wood = num_wood
        self.seed = seed

        # Minimal agent configuration: 1 worker + 1 transporter
        self.agent_config = {
            'scouts': [],  # No scouts for speed
            'workers': [AgentType.WORKER],  # Only 1 worker
            'transporters': [AgentType.TRANSPORTER]  # Only 1 transporter
        }


class UltraFastGameState:
    """Ultra-optimized game state"""

    def __init__(self, grid_size: int, num_obstacles: int):
        self.grid_size = grid_size
        self.base_position = Position(0, 0)
        self.max_steps = 80
        self.current_step = 0
        self.total_score = 0

        # Minimal obstacles
        self.obstacles = self._generate_minimal_obstacles(num_obstacles)

        # Game entities
        self.agents = {}
        self.resources = []

        # Deposited resources
        self.deposited_resources = {ResourceType.GOLD: 0, ResourceType.WOOD: 0}

    def _generate_minimal_obstacles(self, num_obstacles: int) -> set:
        """Generate minimal obstacles"""
        obstacles = set()

        # Fixed obstacle positions for consistency
        if num_obstacles >= 1:
            obstacles.add(Position(3, 3))
        if num_obstacles >= 2:
            obstacles.add(Position(4, 4))

        return obstacles

    def is_valid_position(self, position: Position) -> bool:
        """Fast position validation"""
        return (0 <= position.x < self.grid_size and
                0 <= position.y < self.grid_size and
                position not in self.obstacles)

    def add_agent(self, agent: Agent):
        """Add agent"""
        self.agents[agent.id] = agent

    def add_resource(self, resource: Resource):
        """Add resource"""
        self.resources.append(resource)

    def get_resource_at(self, position: Position) -> Optional[Resource]:
        """Fast resource lookup"""
        for resource in self.resources:
            if resource.is_active and resource.position == position:
                return resource
        return None

    def deposit_resources(self, resources: Dict[ResourceType, int]):
        """Deposit resources"""
        for rtype, amount in resources.items():
            self.deposited_resources[rtype] += amount
            if rtype == ResourceType.GOLD:
                self.total_score += amount * 10
            else:
                self.total_score += amount * 2

    def update(self):
        """Minimal state update"""
        self.current_step += 1

        # Only update resources that might need respawn
        for resource in self.resources:
            if not resource.is_active and resource.respawn_timer > 0:
                resource.respawn_timer -= 1
                if resource.respawn_timer == 0:
                    resource.respawn()

    def is_terminal(self) -> bool:
        """Check terminal condition"""
        return self.current_step >= self.max_steps


class HRGUltraFastEnv:
    """Ultra-fast training version of HRG environment with minimal agents"""

    def __init__(self, config: UltraFastConfig = None):
        self.config = config if config is not None else UltraFastConfig()

        if self.config.seed is not None:
            np.random.seed(self.config.seed)

        # Initialize game state
        self.game_state = UltraFastGameState(
            self.config.grid_size,
            self.config.num_obstacles
        )
        self.game_state.max_steps = self.config.max_steps

        # Create agents and resources
        self.agents = {}
        self.agent_ids = []
        self._create_minimal_agents()
        self._create_simple_resources()

        # Setup spaces
        self._setup_spaces()

        # Episode tracking
        self.episode_count = 0
        self.step_count = 0

        # Pre-allocated arrays for observation
        self._obs_buffer = {agent_id: np.zeros(24, dtype=np.float32)
                           for agent_id in self.agent_ids}

        # Reward system tracking
        self._discovered_resources = set()  # Track discovered resources
        self._agent_discoveries = {agent_id: set() for agent_id in self.agent_ids}
        self._last_deposit_step = {agent_id: -100 for agent_id in self.agent_ids}
        self._steps_without_progress = 0

    def _create_minimal_agents(self):
        """Create minimal agent set"""
        # Only create worker and transporter
        agent_id = "worker_0"
        config = replace(AGENT_CONFIGS[AgentType.WORKER],
                        initial_position=(0, 1),
                        vision_range=2)  # Reduced vision range
        agent = Agent(agent_id, config)
        self.agents[agent_id] = agent
        self.agent_ids.append(agent_id)
        self.game_state.add_agent(agent)

        agent_id = "transporter_0"
        config = replace(AGENT_CONFIGS[AgentType.TRANSPORTER],
                        initial_position=(0, 0),
                        vision_range=3)  # Reduced vision range
        agent = Agent(agent_id, config)
        self.agents[agent_id] = agent
        self.agent_ids.append(agent_id)
        self.game_state.add_agent(agent)

    def _create_simple_resources(self):
        """Create simple resource layout"""
        # Gold in corner
        for _ in range(self.config.num_gold):
            pos = Position(self.config.grid_size - 2, self.config.grid_size - 2)
            if self.game_state.is_valid_position(pos):
                resource = Resource(
                    position=pos,
                    resource_type=ResourceType.GOLD,
                    remaining_quantity=3  # Reduced quantity
                )
                self.game_state.add_resource(resource)

        # Wood distributed simply
        wood_positions = [
            Position(2, 2),
            Position(4, 2),
            Position(2, 4),
            Position(4, 4)
        ]
        for i, pos in enumerate(wood_positions[:self.config.num_wood]):
            if self.game_state.is_valid_position(pos):
                resource = Resource(
                    position=pos,
                    resource_type=ResourceType.WOOD,
                    remaining_quantity=5  # Reduced quantity
                )
                self.game_state.add_resource(resource)

    def _setup_spaces(self):
        """Setup minimal observation and action spaces"""
        self.action_spaces = {agent_id: spaces.Discrete(8) for agent_id in self.agent_ids}

        # Ultra-reduced observation space: 24 dimensions
        self.observation_spaces = {
            agent_id: spaces.Box(low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32)
            for agent_id in self.agent_ids
        }

        self.n_agents = len(self.agent_ids)
        self.agent_ids = sorted(self.agent_ids)
        self.act_dims = {agent_id: 8 for agent_id in self.agent_ids}
        self.obs_dims = {agent_id: 24 for agent_id in self.agent_ids}

    def reset(self) -> Dict[str, np.ndarray]:
        """Reset environment"""
        # Reset game state
        self.game_state = UltraFastGameState(
            self.config.grid_size,
            self.config.num_obstacles
        )
        self.game_state.max_steps = self.config.max_steps

        # Recreate agents and resources
        self.agents.clear()
        self.agent_ids.clear()
        self._create_minimal_agents()
        self._create_simple_resources()

        self.episode_count += 1
        self.step_count = 0

        # Reset reward tracking
        self._discovered_resources.clear()
        self._agent_discoveries = {agent_id: set() for agent_id in self.agent_ids}
        self._last_deposit_step = {agent_id: -100 for agent_id in self.agent_ids}
        self._steps_without_progress = 0

        return {agent_id: self._get_ultra_fast_observation(agent_id)
                for agent_id in self.agent_ids}

    def step(self, actions: Dict[str, Union[int, np.ndarray]]) -> Tuple[
        Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, Any]
    ]:
        """Execute ultra-fast step with improved reward system"""
        self.step_count += 1

        # Reset agent turns
        for agent in self.agents.values():
            agent.reset_turn()

        # Execute actions with enhanced rewards
        step_rewards = {}
        total_resources_before = sum(self.game_state.total_score for _ in [1])  # Track progress

        for agent_id in self.agent_ids:
            if agent_id in actions:
                action = int(actions[agent_id])
                reward = self._execute_action_with_rewards(agent_id, action)
                step_rewards[agent_id] = reward
            else:
                step_rewards[agent_id] = 0.0

        # Update state
        self.game_state.update()

        # Check for progress and apply team rewards
        total_resources_after = self.game_state.total_score
        if total_resources_after > total_resources_before:
            self._steps_without_progress = 0
            team_bonus = 0.5  # Bonus for making progress
        else:
            self._steps_without_progress += 1
            team_bonus = -0.05 if self._steps_without_progress > 10 else 0.01  # Small time penalty

        done = self.game_state.is_terminal()

        # Apply team rewards and time penalty
        final_rewards = {agent_id: step_rewards.get(agent_id, 0.0) + team_bonus
                        for agent_id in self.agent_ids}

        # Get observations
        observations = {agent_id: self._get_ultra_fast_observation(agent_id)
                       for agent_id in self.agent_ids}

        # Enhanced info
        infos = {agent_id: {} for agent_id in self.agent_ids}
        infos['episode'] = {
            'step': self.step_count,
            'total_score': self.game_state.total_score,
            'progress_steps': self._steps_without_progress
        }

        return observations, final_rewards, {agent_id: done for agent_id in self.agent_ids}, infos

    def _execute_action_with_rewards(self, agent_id: str, action: int) -> float:
        """Enhanced action execution with improved reward system"""
        agent = self.agents[agent_id]
        action_type = ActionType(action)

        if not agent.can_perform_action(action_type):
            return -0.1  # Small penalty for invalid actions

        reward = 0.0

        if action_type in [ActionType.MOVE_NORTH, ActionType.MOVE_SOUTH,
                          ActionType.MOVE_WEST, ActionType.MOVE_EAST]:
            reward = self._execute_movement_with_rewards(agent, action_type)

        elif action_type == ActionType.GATHER:
            reward = self._gather_resource_with_rewards(agent)

        elif action_type == ActionType.TRANSFER:
            reward = self._transfer_resources_with_rewards(agent)

        elif action_type == ActionType.DEPOSIT:
            reward = self._deposit_resources_with_rewards(agent)

        elif action_type == ActionType.WAIT:
            reward = -0.02  # Small penalty for waiting

        return reward

    def _execute_movement_with_rewards(self, agent: Agent, action_type: ActionType) -> float:
        """Execute movement with smart rewards"""
        # Calculate new position
        dx = 0 if action_type in [ActionType.MOVE_NORTH, ActionType.MOVE_SOUTH] else (-1 if action_type == ActionType.MOVE_WEST else 1)
        dy = -1 if action_type == ActionType.MOVE_NORTH else (1 if action_type == ActionType.MOVE_SOUTH else 0)
        new_pos = Position(agent.position.x + dx, agent.position.y + dy)

        if not self.game_state.is_valid_position(new_pos):
            return -0.05  # Penalty for invalid movement

        old_pos = agent.position
        agent.position = new_pos
        agent.move_points -= 1.0
        agent.consume_energy(agent.config.energy_consumption_move)

        reward = 0.0

        # Discovery rewards
        resource_at_new_pos = self.game_state.get_resource_at(new_pos)
        if resource_at_new_pos and resource_at_new_pos.is_active:
            resource_id = (resource_at_new_pos.position.x, resource_at_new_pos.position.y)
            if resource_id not in self._agent_discoveries[agent.id]:
                self._agent_discoveries[agent.id].add(resource_id)
                self._discovered_resources.add(resource_id)
                reward += 2.0 if resource_at_new_pos.resource_type == ResourceType.GOLD else 1.0

        # Check for discovering other agent
        for other_agent in self.agents.values():
            if other_agent.id != agent.id and other_agent.position == new_pos:
                reward += 0.5

        # Directional movement rewards
        if agent.type == AgentType.WORKER:
            # Worker: reward for moving towards resources
            nearest_resource_dist = self._get_distance_to_nearest_resource(agent.position)
            if nearest_resource_dist < 10:
                reward += 0.1 * (1.0 / max(nearest_resource_dist, 1))

        elif agent.type == AgentType.TRANSPORTER:
            # Transporter: reward for strategic positioning
            if agent.is_carrying_resources:
                # Move towards base when carrying resources
                base_dist = agent.position.distance_to(self.game_state.base_position)
                reward += 0.2 * (1.0 / max(base_dist, 1))
            else:
                # Move towards worker when empty
                for other_agent in self.agents.values():
                    if other_agent.type == AgentType.WORKER and other_agent.is_carrying_resources:
                        dist = agent.position.distance_to(other_agent.position)
                        reward += 0.1 * (1.0 / max(dist, 1))
                        break

        # Small movement cost
        reward -= 0.01

        return reward

    def _gather_resource_with_rewards(self, agent: Agent) -> float:
        """Enhanced resource gathering with better rewards"""
        if agent.type == AgentType.SCOUT:
            return 0.0

        resource = self.game_state.get_resource_at(agent.position)
        if not resource or not resource.is_active:
            return -0.1  # Penalty for trying to gather where there's nothing

        # Starting to gather
        if agent.gathering_target != resource:
            agent.gathering_target = resource
            agent.gathering_progress = 0
            agent.consume_energy(agent.config.energy_consumption_gather)
            return 0.5  # Reward for starting to gather

        # Continuing to gather
        agent.gathering_progress += 1
        agent.consume_energy(agent.config.energy_consumption_gather)

        required_time = 1 if resource.resource_type == ResourceType.WOOD else 2
        if agent.gathering_progress >= required_time:
            gathered = resource.gather()
            if gathered > 0:
                agent.add_resources(resource.resource_type, gathered)
                base_reward = RESOURCE_CONFIGS[resource.resource_type].value
                reward = base_reward * 1.0  # Full reward for completion
                agent.gathering_progress = 0
                agent.gathering_target = None
                return reward

        return 0.1  # Small reward for making progress

    def _transfer_resources_with_rewards(self, agent: Agent) -> float:
        """Enhanced resource transfer with better coordination rewards"""
        if not agent.is_carrying_resources:
            return -0.1  # Penalty for trying to transfer when empty

        # Find adjacent agents
        for other_agent in self.agents.values():
            if other_agent.id != agent.id and agent.position.is_adjacent(other_agent.position):
                if agent.type == AgentType.WORKER and other_agent.type == AgentType.TRANSPORTER:
                    if other_agent.can_carry_more:
                        # Calculate transfer value
                        transferred_value = 0.0
                        for rtype in ResourceType:
                            if agent.inventory[rtype] > 0:
                                amount = agent.remove_resources(rtype, agent.inventory[rtype])
                                added = other_agent.add_resources(rtype, amount)
                                transferred_value += added * RESOURCE_CONFIGS[rtype].value * 0.5

                        agent.consume_energy(agent.config.energy_consumption_transfer)

                        # Bonus reward for successful coordination
                        return transferred_value + 1.0
                    else:
                        return -0.05  # Small penalty if transporter is full

                elif agent.type == AgentType.TRANSPORTER and other_agent.type == AgentType.WORKER:
                    # This shouldn't happen in normal workflow, but handle it
                    return -0.05

        return -0.1  # Penalty for trying to transfer when no valid target

    def _deposit_resources_with_rewards(self, agent: Agent) -> float:
        """Enhanced resource deposit with better rewards"""
        if not agent.is_at_base:
            return -0.1  # Penalty for trying to deposit away from base

        if not agent.is_carrying_resources:
            return -0.05  # Small penalty for trying to deposit when empty

        deposited_resources = agent.clear_inventory()
        self.game_state.deposit_resources(deposited_resources)

        # Calculate deposit rewards
        reward = 0.0
        for rtype, amount in deposited_resources.items():
            base_value = RESOURCE_CONFIGS[rtype].value
            if rtype == ResourceType.GOLD:
                reward += amount * base_value * 1.5  # Higher reward for gold
            else:
                reward += amount * base_value * 1.0  # Standard reward for wood

        # Bonus for successful deposit
        reward += 2.0

        # Track deposit for progress tracking
        self._last_deposit_step[agent.id] = self.step_count

        return reward

    def _get_distance_to_nearest_resource(self, position: Position) -> int:
        """Get Manhattan distance to nearest active resource"""
        min_dist = float('inf')
        for resource in self.game_state.resources:
            if resource.is_active:
                dist = position.distance_to(resource.position)
                min_dist = min(min_dist, dist)
        return min_dist if min_dist != float('inf') else 999

    
    def _get_ultra_fast_observation(self, agent_id: str) -> np.ndarray:
        """Ultra-fast observation - reduced to 24 dimensions"""
        agent = self.agents[agent_id]
        obs = self._obs_buffer[agent_id]
        obs.fill(0)

        idx = 0
        grid_size_f = float(self.config.grid_size)

        # Agent self state (8 dimensions)
        obs[idx] = agent.position.x / grid_size_f
        obs[idx + 1] = agent.position.y / grid_size_f
        obs[idx + 2] = 1.0 if agent.type == AgentType.WORKER else 0.0
        obs[idx + 3] = 1.0 if agent.type == AgentType.TRANSPORTER else 0.0
        obs[idx + 4] = agent.inventory[ResourceType.GOLD] / 5.0
        obs[idx + 5] = agent.inventory[ResourceType.WOOD] / 10.0
        obs[idx + 6] = agent.energy / 100.0
        obs[idx + 7] = 1.0 - (self.game_state.current_step / self.game_state.max_steps)
        idx += 8

        # Resource locations (16 dimensions for 8 resources * 2 dims each)
        max_resources = 8
        resource_count = 0
        vision_range = 4  # Extended vision for better learning

        for resource in self.game_state.resources:
            if resource_count >= max_resources:
                break
            if resource.is_active:
                dx = resource.position.x - agent.position.x
                dy = resource.position.y - agent.position.y

                # Only include if within vision range
                if abs(dx) <= vision_range and abs(dy) <= vision_range:
                    obs[idx] = dx / vision_range
                    obs[idx + 1] = dy / vision_range
                    idx += 2
                    resource_count += 1

        return obs

    def get_global_state(self) -> np.ndarray:
        """Ultra-fast global state"""
        # With 2 agents, state is 22 dimensions: 2*8 + 6
        state = np.zeros(22, dtype=np.float32)
        idx = 0
        grid_size_f = float(self.config.grid_size)

        # Agent states (16 dimensions)
        for agent_id in sorted(self.agent_ids):
            agent = self.agents[agent_id]
            state[idx] = agent.position.x / grid_size_f
            state[idx + 1] = agent.position.y / grid_size_f
            state[idx + 2] = 1.0 if agent.type == AgentType.WORKER else 0.0
            state[idx + 3] = 1.0 if agent.type == AgentType.TRANSPORTER else 0.0
            state[idx + 4] = agent.inventory[ResourceType.GOLD] / 5.0
            state[idx + 5] = agent.inventory[ResourceType.WOOD] / 10.0
            state[idx + 6] = agent.energy / 100.0
            state[idx + 7] = 1.0 if agent.is_at_base else 0.0
            idx += 8

        # Resource summary (6 dimensions)
        active_gold = sum(1 for r in self.game_state.resources
                         if r.is_active and r.resource_type == ResourceType.GOLD)
        active_wood = sum(1 for r in self.game_state.resources
                         if r.is_active and r.resource_type == ResourceType.WOOD)

        state[idx] = active_gold / 5.0
        state[idx + 1] = active_wood / 10.0
        state[idx + 2] = self.game_state.total_score / 500.0
        state[idx + 3] = self.game_state.deposited_resources[ResourceType.GOLD] / 10.0
        state[idx + 4] = self.game_state.deposited_resources[ResourceType.WOOD] / 20.0
        state[idx + 5] = self.game_state.current_step / self.game_state.max_steps

        return state

    def get_avail_actions(self, agent_id: str) -> List[int]:
        """Get available actions"""
        agent = self.agents[agent_id]
        return [action.value for action in ActionType
                if agent.can_perform_action(action)]

    def close(self):
        """Close environment"""
        pass

    def render(self, mode='human'):
        """No rendering in ultra-fast version"""
        pass


# Factory function
def create_hrg_ultra_fast_env(**kwargs) -> HRGUltraFastEnv:
    """Create ultra-fast training environment"""
    config = UltraFastConfig(**kwargs)
    return HRGUltraFastEnv(config)