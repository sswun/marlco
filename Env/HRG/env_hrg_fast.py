"""
HRG (Heterogeneous Resource Gathering) Environment - Fast Training Version

Optimized for maximum training speed while maintaining environment correctness.
Key optimizations:
- Simplified observation space
- Optimized state lookups using spatial hashing
- Removed rendering overhead
- Streamlined reward calculations
- Efficient resource management
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


class HRGFastConfig:
    """Fast training configuration"""

    def __init__(self,
                 grid_size: int = 8,
                 max_steps: int = 100,
                 num_obstacles: int = 5,
                 num_gold: int = 2,
                 num_wood: int = 8,
                 agent_config: Dict[str, List[AgentType]] = None,
                 seed: Optional[int] = None):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.num_obstacles = num_obstacles
        self.num_gold = num_gold
        self.num_wood = num_wood
        self.seed = seed

        if agent_config is None:
            self.agent_config = {
                'scouts': [AgentType.SCOUT, AgentType.SCOUT],
                'workers': [AgentType.WORKER, AgentType.WORKER, AgentType.WORKER],
                'transporters': [AgentType.TRANSPORTER]
            }
        else:
            self.agent_config = agent_config


class FastGameState:
    """Optimized game state with spatial indexing"""

    def __init__(self, grid_size: int, num_obstacles: int):
        self.grid_size = grid_size
        self.base_position = Position(0, 0)
        self.max_steps = 100
        self.current_step = 0
        self.total_score = 0

        # Spatial index for fast lookups
        self.position_grid = {}  # position -> list of entity IDs
        
        # Obstacles
        self.obstacles = self._generate_obstacles_fast(num_obstacles)
        
        # Game entities
        self.agents = {}
        self.resources = []
        self.active_resources = set()  # Fast lookup for active resources
        
        # Deposited resources
        self.deposited_resources = {ResourceType.GOLD: 0, ResourceType.WOOD: 0}
        
        # Message history (minimal)
        self.message_history = []

    def _generate_obstacles_fast(self, num_obstacles: int) -> set:
        """Fast obstacle generation"""
        obstacles = set()
        np.random.seed(None)
        
        attempts = 0
        while len(obstacles) < num_obstacles and attempts < num_obstacles * 3:
            x = np.random.randint(2, self.grid_size)
            y = np.random.randint(2, self.grid_size)
            pos = Position(x, y)
            
            if pos not in obstacles:
                obstacles.add(pos)
            attempts += 1
        
        return obstacles

    def is_valid_position(self, position: Position) -> bool:
        """Fast position validation"""
        return (0 <= position.x < self.grid_size and
                0 <= position.y < self.grid_size and
                position not in self.obstacles)

    def add_agent(self, agent: Agent):
        """Add agent with spatial indexing"""
        self.agents[agent.id] = agent

    def add_resource(self, resource: Resource):
        """Add resource with spatial indexing"""
        self.resources.append(resource)
        if resource.is_active:
            self.active_resources.add(id(resource))

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
                    self.active_resources.add(id(resource))

    def is_terminal(self) -> bool:
        """Check terminal condition"""
        return self.current_step >= self.max_steps


class HRGFastEnv:
    """Fast training version of HRG environment"""

    def __init__(self, config: HRGFastConfig = None):
        self.config = config if config is not None else HRGFastConfig()

        if self.config.seed is not None:
            np.random.seed(self.config.seed)

        # Initialize game state
        self.game_state = FastGameState(
            self.config.grid_size,
            self.config.num_obstacles
        )
        self.game_state.max_steps = self.config.max_steps

        # Create agents and resources
        self.agents = {}
        self.agent_ids = []
        self._create_agents()
        self._create_resources()

        # Setup spaces
        self._setup_spaces()

        # Episode tracking
        self.episode_count = 0
        self.step_count = 0

        # Pre-allocated arrays for observation
        self._obs_buffer = {agent_id: np.zeros(40, dtype=np.float32) 
                           for agent_id in self.agent_ids}

    def _create_agents(self):
        """Create agents"""
        for i, agent_type in enumerate(self.config.agent_config.get('scouts', [])):
            agent_id = f"scout_{i}"
            config = replace(AGENT_CONFIGS[agent_type], initial_position=self._get_safe_position())
            agent = Agent(agent_id, config)
            self.agents[agent_id] = agent
            self.agent_ids.append(agent_id)
            self.game_state.add_agent(agent)

        for i, agent_type in enumerate(self.config.agent_config.get('workers', [])):
            agent_id = f"worker_{i}"
            config = replace(AGENT_CONFIGS[agent_type], initial_position=self._get_safe_position())
            agent = Agent(agent_id, config)
            self.agents[agent_id] = agent
            self.agent_ids.append(agent_id)
            self.game_state.add_agent(agent)

        for i, agent_type in enumerate(self.config.agent_config.get('transporters', [])):
            agent_id = f"transporter_{i}"
            config = replace(AGENT_CONFIGS[agent_type], initial_position=(0, 0))
            agent = Agent(agent_id, config)
            self.agents[agent_id] = agent
            self.agent_ids.append(agent_id)
            self.game_state.add_agent(agent)

    def _get_safe_position(self) -> Tuple[int, int]:
        """Get safe starting position near base"""
        safe_positions = [(0, 1), (1, 0), (1, 1)]
        for pos in safe_positions:
            if self.game_state.is_valid_position(Position(*pos)):
                return pos
        return (0, 1)

    def _create_resources(self):
        """Create resources efficiently"""
        # Gold in far corner
        for _ in range(self.config.num_gold):
            x = np.random.randint(self.config.grid_size - 3, self.config.grid_size)
            y = np.random.randint(self.config.grid_size - 3, self.config.grid_size)
            pos = Position(x, y)
            if self.game_state.is_valid_position(pos):
                resource = Resource(
                    position=pos,
                    resource_type=ResourceType.GOLD,
                    remaining_quantity=RESOURCE_CONFIGS[ResourceType.GOLD].quantity_per_unit
                )
                self.game_state.add_resource(resource)

        # Wood distributed
        for _ in range(self.config.num_wood):
            x = np.random.randint(2, self.config.grid_size)
            y = np.random.randint(2, self.config.grid_size)
            pos = Position(x, y)
            if self.game_state.is_valid_position(pos):
                resource = Resource(
                    position=pos,
                    resource_type=ResourceType.WOOD,
                    remaining_quantity=RESOURCE_CONFIGS[ResourceType.WOOD].quantity_per_unit
                )
                self.game_state.add_resource(resource)

    def _setup_spaces(self):
        """Setup minimal observation and action spaces"""
        self.action_spaces = {agent_id: spaces.Discrete(8) for agent_id in self.agent_ids}
        
        # Reduced observation space: 40 dimensions
        self.observation_spaces = {
            agent_id: spaces.Box(low=-np.inf, high=np.inf, shape=(40,), dtype=np.float32)
            for agent_id in self.agent_ids
        }

        self.n_agents = len(self.agent_ids)
        self.agent_ids = sorted(self.agent_ids)
        self.act_dims = {agent_id: 8 for agent_id in self.agent_ids}
        self.obs_dims = {agent_id: 40 for agent_id in self.agent_ids}

    def reset(self) -> Dict[str, np.ndarray]:
        """Reset environment"""
        # Reset game state
        self.game_state = FastGameState(
            self.config.grid_size,
            self.config.num_obstacles
        )
        self.game_state.max_steps = self.config.max_steps

        # Recreate agents and resources
        self.agents.clear()
        self.agent_ids.clear()  # Clear agent IDs to prevent accumulation
        self._create_agents()
        self._create_resources()

        self.episode_count += 1
        self.step_count = 0

        return {agent_id: self._get_observation_fast(agent_id)
                for agent_id in self.agent_ids}

    def step(self, actions: Dict[str, Union[int, np.ndarray]]) -> Tuple[
        Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, Any]
    ]:
        """Execute step - optimized version"""
        self.step_count += 1

        # Reset agent turns
        for agent in self.agents.values():
            agent.reset_turn()

        # Execute actions (no shuffling for speed)
        step_rewards = {}
        for agent_id in self.agent_ids:
            if agent_id in actions:
                action = int(actions[agent_id])
                reward = self._execute_action_fast(agent_id, action)
                step_rewards[agent_id] = reward
            else:
                step_rewards[agent_id] = 0.0

        # Update state
        self.game_state.update()

        # Calculate rewards
        team_reward = -0.01  # Time penalty only
        done = self.game_state.is_terminal()

        final_rewards = {agent_id: step_rewards.get(agent_id, 0.0) + team_reward
                        for agent_id in self.agent_ids}

        # Get observations
        observations = {agent_id: self._get_observation_fast(agent_id)
                       for agent_id in self.agent_ids}

        # Minimal info
        infos = {agent_id: {} for agent_id in self.agent_ids}
        infos['episode'] = {
            'step': self.step_count,
            'total_score': self.game_state.total_score
        }

        return observations, final_rewards, {agent_id: done for agent_id in self.agent_ids}, infos

    def _execute_action_fast(self, agent_id: str, action: int) -> float:
        """Fast action execution"""
        agent = self.agents[agent_id]
        action_type = ActionType(action)

        if not agent.can_perform_action(action_type):
            return 0.0

        reward = 0.0

        if action_type in [ActionType.MOVE_NORTH, ActionType.MOVE_SOUTH, 
                          ActionType.MOVE_WEST, ActionType.MOVE_EAST]:
            # Movement
            dx = 0 if action_type in [ActionType.MOVE_NORTH, ActionType.MOVE_SOUTH] else (-1 if action_type == ActionType.MOVE_WEST else 1)
            dy = -1 if action_type == ActionType.MOVE_NORTH else (1 if action_type == ActionType.MOVE_SOUTH else 0)
            
            if agent.move_points >= 1.0:
                new_pos = Position(agent.position.x + dx, agent.position.y + dy)
                if self.game_state.is_valid_position(new_pos):
                    agent.position = new_pos
                    agent.move_points -= 1.0
                    agent.consume_energy(agent.config.energy_consumption_move)

        elif action_type == ActionType.GATHER:
            reward = self._gather_resource_fast(agent)

        elif action_type == ActionType.TRANSFER:
            reward = self._transfer_resources_fast(agent)

        elif action_type == ActionType.DEPOSIT:
            reward = self._deposit_resources_fast(agent)

        return reward

    def _gather_resource_fast(self, agent: Agent) -> float:
        """Fast resource gathering"""
        if agent.type == AgentType.SCOUT:
            return 0.0

        resource = self.game_state.get_resource_at(agent.position)
        if not resource or not resource.is_active:
            return 0.0

        if agent.gathering_target == resource:
            agent.gathering_progress += 1
            agent.consume_energy(agent.config.energy_consumption_gather)

            required_time = RESOURCE_CONFIGS[resource.resource_type].gather_difficulty
            if agent.gathering_progress >= required_time:
                gathered = resource.gather()
                if gathered > 0:
                    agent.add_resources(resource.resource_type, gathered)
                    reward = RESOURCE_CONFIGS[resource.resource_type].value * 0.1
                    agent.gathering_target = None
                    agent.gathering_progress = 0
                    if not resource.is_active:
                        self.game_state.active_resources.discard(id(resource))
                    return reward
        else:
            agent.gathering_target = resource
            agent.gathering_progress = 0
            agent.consume_energy(agent.config.energy_consumption_gather)

        return 0.0

    def _transfer_resources_fast(self, agent: Agent) -> float:
        """Fast resource transfer"""
        if not agent.is_carrying_resources:
            return 0.0

        # Find adjacent agents
        target_agent = None
        for other_agent in self.agents.values():
            if (other_agent.id != agent.id and 
                agent.position.is_adjacent(other_agent.position)):
                if agent.type == AgentType.WORKER and other_agent.type == AgentType.TRANSPORTER:
                    target_agent = other_agent
                    break
                elif agent.type == AgentType.TRANSPORTER and other_agent.type == AgentType.WORKER:
                    target_agent = other_agent
                    break

        if not target_agent or not target_agent.can_carry_more:
            return 0.0

        # Transfer
        transferred_value = 0.0
        for rtype in ResourceType:
            if agent.inventory[rtype] > 0:
                amount = agent.remove_resources(rtype, agent.inventory[rtype])
                added = target_agent.add_resources(rtype, amount)
                transferred_value += added * RESOURCE_CONFIGS[rtype].value * 0.05

        agent.consume_energy(agent.config.energy_consumption_transfer)
        return transferred_value

    def _deposit_resources_fast(self, agent: Agent) -> float:
        """Fast resource deposit"""
        if not agent.is_at_base or not agent.is_carrying_resources:
            return 0.0

        deposited_resources = agent.clear_inventory()
        self.game_state.deposit_resources(deposited_resources)

        reward = 0.0
        for rtype, amount in deposited_resources.items():
            reward += amount * RESOURCE_CONFIGS[rtype].value * 0.5

        return reward

    def _get_observation_fast(self, agent_id: str) -> np.ndarray:
        """Ultra-fast observation - reduced to 40 dimensions"""
        agent = self.agents[agent_id]
        obs = self._obs_buffer[agent_id]
        obs.fill(0)
        
        idx = 0
        grid_size_f = float(self.config.grid_size)

        # Agent self state (10 dimensions)
        obs[idx] = agent.position.x / grid_size_f
        obs[idx + 1] = agent.position.y / grid_size_f
        obs[idx + 2 + agent.type.value] = 1.0
        obs[idx + 5] = agent.inventory[ResourceType.GOLD] / 10.0
        obs[idx + 6] = agent.inventory[ResourceType.WOOD] / 10.0
        obs[idx + 7] = agent.energy / 100.0
        obs[idx + 8] = agent.position.distance_to(self.game_state.base_position) / (grid_size_f * 2)
        obs[idx + 9] = 1.0 - (self.game_state.current_step / self.game_state.max_steps)
        idx += 10

        # Nearby entities (30 dimensions for 6 entities * 5 dims each)
        vision_range = 3
        entity_count = 0
        max_entities = 6

        for dx in range(-vision_range, vision_range + 1):
            if entity_count >= max_entities:
                break
            for dy in range(-vision_range, vision_range + 1):
                if entity_count >= max_entities:
                    break
                if dx == 0 and dy == 0:
                    continue

                check_pos = Position(agent.position.x + dx, agent.position.y + dy)
                if not (0 <= check_pos.x < self.config.grid_size and 
                        0 <= check_pos.y < self.config.grid_size):
                    continue

                # Check agents first
                found = False
                for other_agent in self.agents.values():
                    if other_agent.id != agent_id and other_agent.position == check_pos:
                        obs[idx] = dx / vision_range
                        obs[idx + 1] = dy / vision_range
                        obs[idx + 2 + other_agent.type.value] = 1.0
                        idx += 5
                        entity_count += 1
                        found = True
                        break

                if not found:
                    # Check resources
                    resource = self.game_state.get_resource_at(check_pos)
                    if resource and resource.is_active:
                        obs[idx] = dx / vision_range
                        obs[idx + 1] = dy / vision_range
                        obs[idx + 2 + resource.resource_type.value] = 1.0
                        obs[idx + 4] = resource.remaining_quantity / 5.0
                        idx += 5
                        entity_count += 1

        return obs

    def get_global_state(self) -> np.ndarray:
        """Fast global state - dynamically sized"""
        # Calculate required dimensions
        n_agents = len(self.agent_ids)
        state_dim = n_agents * 8 + 6  # Each agent: 8 dims, plus 6 for resources

        # Create fresh state array to prevent accumulation
        state = np.zeros(state_dim, dtype=np.float32)
        idx = 0
        grid_size_f = float(self.config.grid_size)

        # Agent states (n_agents * 8 dims)
        for agent_id in sorted(self.agent_ids):
            agent = self.agents[agent_id]
            if idx + 8 <= state_dim:  # Bounds check
                state[idx] = agent.position.x / grid_size_f
                state[idx + 1] = agent.position.y / grid_size_f
                # Type one-hot (3 dimensions)
                state[idx + 2] = 1.0 if agent.type == AgentType.SCOUT else 0.0
                state[idx + 3] = 1.0 if agent.type == AgentType.WORKER else 0.0
                state[idx + 4] = 1.0 if agent.type == AgentType.TRANSPORTER else 0.0
                # Inventory and energy
                state[idx + 5] = agent.inventory[ResourceType.GOLD] / 10.0
                state[idx + 6] = agent.inventory[ResourceType.WOOD] / 10.0
                state[idx + 7] = agent.energy / 100.0
                idx += 8

        # Resource summary (6 dimensions)
        if idx + 6 <= state_dim:  # Bounds check
            active_gold = sum(1 for r in self.game_state.resources
                             if r.is_active and r.resource_type == ResourceType.GOLD)
            active_wood = sum(1 for r in self.game_state.resources
                             if r.is_active and r.resource_type == ResourceType.WOOD)

            state[idx] = active_gold / 10.0
            state[idx + 1] = active_wood / 20.0
            state[idx + 2] = self.game_state.total_score / 1000.0
            state[idx + 3] = self.game_state.deposited_resources[ResourceType.GOLD] / 20.0
            state[idx + 4] = self.game_state.deposited_resources[ResourceType.WOOD] / 50.0
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
        """No rendering in fast version"""
        pass


# Factory function
def create_hrg_fast_env(difficulty: str = "fast_training", **kwargs) -> HRGFastEnv:
    """Create fast training environment"""
    if difficulty == "ultra_fast":
        config = HRGFastConfig(
            grid_size=6,
            max_steps=80,
            num_obstacles=3,
            num_gold=1,
            num_wood=5,
            **kwargs
        )
    else:
        config = HRGFastConfig(**kwargs)
    
    return HRGFastEnv(config)
