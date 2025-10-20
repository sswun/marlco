"""
HRG (Heterogeneous Resource Gathering) Environment

A multi-agent reinforcement learning environment where agents with different
roles work together to collect resources efficiently.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import random
from gymnasium import spaces

from .core import (
    Agent, AgentType, ResourceType, ActionType, Position, Resource,
    GameState, AGENT_CONFIGS, RESOURCE_CONFIGS
)

# Configure logging
logger = logging.getLogger(__name__)


class HRGConfig:
    """Configuration class for HRG environment"""

    def __init__(self,
                 grid_size: int = 10,
                 max_steps: int = 200,
                 num_obstacles: int = 10,
                 num_gold: int = 3,
                 num_wood: int = 10,
                 agent_config: Dict[str, List[AgentType]] = None,
                 render_mode: str = "rgb_array",
                 render_fps: int = 4,
                 seed: Optional[int] = None):
        """
        Initialize HRG configuration

        Args:
            grid_size: Size of the square grid
            max_steps: Maximum number of steps per episode
            num_obstacles: Number of obstacles on the grid
            num_gold: Number of gold resources
            num_wood: Number of wood resources
            agent_config: Dictionary specifying agent types and their initial positions
            render_mode: Rendering mode ("human", "rgb_array", or None)
            render_fps: Rendering frames per second
            seed: Random seed for reproducibility
        """
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.num_obstacles = num_obstacles
        self.num_gold = num_gold
        self.num_wood = num_wood
        self.render_mode = render_mode
        self.render_fps = render_fps
        self.seed = seed

        # Default agent configuration
        if agent_config is None:
            self.agent_config = {
                'scouts': [AgentType.SCOUT, AgentType.SCOUT],
                'workers': [AgentType.WORKER, AgentType.WORKER, AgentType.WORKER],
                'transporters': [AgentType.TRANSPORTER]
            }
        else:
            self.agent_config = agent_config


class HRGEnv:
    """
    Heterogeneous Resource Gathering Environment

    A multi-agent environment where agents with different capabilities
    collaborate to collect resources and bring them back to base.
    """

    def __init__(self, config: HRGConfig = None):
        """
        Initialize HRG environment

        Args:
            config: Environment configuration
        """
        self.config = config if config is not None else HRGConfig()

        # Set random seed
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
            random.seed(self.config.seed)

        # Initialize game state
        self.game_state = GameState(
            grid_size=self.config.grid_size,
            num_obstacles=self.config.num_obstacles
        )
        self.game_state.max_steps = self.config.max_steps

        # Initialize agents
        self.agents = {}
        self.agent_ids = []
        self._create_agents()

        # Initialize resources
        self._create_resources()

        # Observation and action spaces
        self._setup_spaces()

        # Rendering
        self.renderer = None
        self._setup_renderer()

        # Episode tracking
        self.episode_count = 0
        self.step_count = 0

        logger.info(f"HRG environment initialized with {len(self.agents)} agents")

    def _create_agents(self):
        """Create agents based on configuration"""
        agent_id_counter = 0

        # Create scouts
        for i, agent_type in enumerate(self.config.agent_config.get('scouts', [])):
            agent_id = f"scout_{i}"
            config = AGENT_CONFIGS[agent_type]
            # Randomize initial position near base
            initial_pos = self._get_random_base_adjacent_position()
            config.initial_position = initial_pos
            agent = Agent(agent_id, config)
            self.agents[agent_id] = agent
            self.agent_ids.append(agent_id)
            agent_id_counter += 1

        # Create workers
        for i, agent_type in enumerate(self.config.agent_config.get('workers', [])):
            agent_id = f"worker_{i}"
            config = AGENT_CONFIGS[agent_type]
            initial_pos = self._get_random_base_adjacent_position()
            config.initial_position = initial_pos
            agent = Agent(agent_id, config)
            self.agents[agent_id] = agent
            self.agent_ids.append(agent_id)
            agent_id_counter += 1

        # Create transporters
        for i, agent_type in enumerate(self.config.agent_config.get('transporters', [])):
            agent_id = f"transporter_{i}"
            config = AGENT_CONFIGS[agent_type]
            # Transporter always starts at base
            config.initial_position = (0, 0)
            agent = Agent(agent_id, config)
            self.agents[agent_id] = agent
            self.agent_ids.append(agent_id)
            agent_id_counter += 1

        # Add agents to game state
        for agent in self.agents.values():
            self.game_state.add_agent(agent)

    def _get_random_base_adjacent_position(self) -> Tuple[int, int]:
        """Get a random position adjacent to base"""
        adjacent_positions = [(0, 1), (1, 0), (1, 1), (0, 2), (2, 0)]
        available = [pos for pos in adjacent_positions
                    if self.game_state.is_valid_position(Position(*pos))]
        return random.choice(available) if available else (1, 0)

    def _create_resources(self):
        """Create resources on the grid"""
        # Create gold resources (clustered in far corner)
        gold_positions = self._generate_clustered_positions(
            self.config.num_gold,
            center_x=self.config.grid_size - 2,
            center_y=self.config.grid_size - 2,
            radius=2
        )

        for pos in gold_positions:
            resource = Resource(
                position=pos,
                resource_type=ResourceType.GOLD,
                remaining_quantity=RESOURCE_CONFIGS[ResourceType.GOLD].quantity_per_unit
            )
            self.game_state.add_resource(resource)

        # Create wood resources (distributed)
        wood_positions = self._generate_distributed_positions(
            self.config.num_wood,
            min_distance=2
        )

        for pos in wood_positions:
            resource = Resource(
                position=pos,
                resource_type=ResourceType.WOOD,
                remaining_quantity=RESOURCE_CONFIGS[ResourceType.WOOD].quantity_per_unit
            )
            self.game_state.add_resource(resource)

    def _generate_clustered_positions(self, count: int, center_x: int, center_y: int, radius: int) -> List[Position]:
        """Generate positions clustered around a center point"""
        positions = []
        attempts = 0

        while len(positions) < count and attempts < count * 10:
            x = center_x + random.randint(-radius, radius)
            y = center_y + random.randint(-radius, radius)
            pos = Position(x, y)

            if (self.game_state.is_valid_position(pos) and
                pos not in positions and
                not self.game_state.get_resource_at(pos)):
                positions.append(pos)

            attempts += 1

        return positions

    def _generate_distributed_positions(self, count: int, min_distance: int = 2) -> List[Position]:
        """Generate distributed positions with minimum distance"""
        positions = []
        attempts = 0

        while len(positions) < count and attempts < count * 10:
            x = random.randint(2, self.config.grid_size - 1)
            y = random.randint(2, self.config.grid_size - 1)
            pos = Position(x, y)

            # Check minimum distance from other positions
            valid = (self.game_state.is_valid_position(pos) and
                    pos not in positions and
                    not self.game_state.get_resource_at(pos))

            if valid:
                for other_pos in positions:
                    if pos.distance_to(other_pos) < min_distance:
                        valid = False
                        break

            if valid:
                positions.append(pos)

            attempts += 1

        return positions

    def _setup_spaces(self):
        """Setup observation and action spaces"""
        # Action space: 8 discrete actions for all agents
        self.action_spaces = {
            agent_id: spaces.Discrete(8) for agent_id in self.agent_ids
        }

        # Observation space: reduced to 60 dimensions per agent for performance
        self.observation_spaces = {
            agent_id: spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(60,), dtype=np.float32
            ) for agent_id in self.agent_ids
        }

        # Dimensions for compatibility
        self.n_agents = len(self.agent_ids)
        self.agent_ids = sorted(self.agent_ids)
        self.act_dims = {agent_id: 8 for agent_id in self.agent_ids}
        self.obs_dims = {agent_id: 60 for agent_id in self.agent_ids}

    def _setup_renderer(self):
        """Setup renderer based on render mode"""
        if self.config.render_mode in ["human", "rgb_array"]:
            from .renderer import HRGRenderer
            self.renderer = HRGRenderer(self.config.grid_size)

    def reset(self) -> Dict[str, np.ndarray]:
        """
        Reset environment and return initial observations

        Returns:
            Dict[str, np.ndarray]: Initial observations for each agent
        """
        # Reset game state
        self.game_state = GameState(
            grid_size=self.config.grid_size,
            num_obstacles=self.config.num_obstacles
        )
        self.game_state.max_steps = self.config.max_steps

        # Recreate agents and resources
        self.agents.clear()
        self._create_agents()
        self._create_resources()

        # Reset episode tracking
        self.episode_count += 1
        self.step_count = 0

        # Return initial observations
        observations = {}
        for agent_id in self.agent_ids:
            observations[agent_id] = self._get_observation(agent_id)

        return observations

    def step(self, actions: Dict[str, Union[int, np.ndarray]]) -> Tuple[
        Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, Any]
    ]:
        """
        Execute one step of the environment

        Args:
            actions: Dictionary of actions for each agent

        Returns:
            Tuple of (observations, rewards, dones, infos)
        """
        self.step_count += 1

        # Reset agent turn states
        for agent in self.agents.values():
            agent.reset_turn()

        # Execute actions in random order to avoid bias
        agent_order = list(self.agent_ids)
        random.shuffle(agent_order)

        step_rewards = {}
        for agent_id in agent_order:
            if agent_id in actions:
                action = int(actions[agent_id])
                reward = self._execute_action(agent_id, action)
                step_rewards[agent_id] = reward
            else:
                step_rewards[agent_id] = 0.0

        # Update game state
        self.game_state.update()

        # Calculate rewards and check termination
        team_reward = self._calculate_team_reward()
        done = self.game_state.is_terminal()

        # Distribute team reward to all agents
        final_rewards = {agent_id: step_rewards.get(agent_id, 0.0) + team_reward
                        for agent_id in self.agent_ids}

        # Get observations
        observations = {agent_id: self._get_observation(agent_id)
                       for agent_id in self.agent_ids}

        # Create info dictionary
        infos = {agent_id: self._get_info(agent_id) for agent_id in self.agent_ids}
        infos['episode'] = {
            'step': self.step_count,
            'total_score': self.game_state.total_score,
            'gold_deposited': self.game_state.deposited_resources[ResourceType.GOLD],
            'wood_deposited': self.game_state.deposited_resources[ResourceType.WOOD],
        }

        # Handle rendering
        if self.renderer is not None:
            self._render_step()

        return observations, final_rewards, {agent_id: done for agent_id in self.agent_ids}, infos

    def _execute_action(self, agent_id: str, action: int) -> float:
        """Execute action for a specific agent"""
        agent = self.agents[agent_id]
        action_type = ActionType(action)

        if not agent.can_perform_action(action_type):
            return 0.0  # Small penalty for invalid action

        reward = 0.0

        if action_type == ActionType.MOVE_NORTH:
            reward = self._move_agent(agent, 0, -1)
        elif action_type == ActionType.MOVE_SOUTH:
            reward = self._move_agent(agent, 0, 1)
        elif action_type == ActionType.MOVE_WEST:
            reward = self._move_agent(agent, -1, 0)
        elif action_type == ActionType.MOVE_EAST:
            reward = self._move_agent(agent, 1, 0)
        elif action_type == ActionType.GATHER:
            reward = self._gather_resource(agent)
        elif action_type == ActionType.TRANSFER:
            reward = self._transfer_resources(agent)
        elif action_type == ActionType.DEPOSIT:
            reward = self._deposit_resources(agent)
        elif action_type == ActionType.WAIT:
            reward = 0.0  # No reward for waiting

        return reward

    def _move_agent(self, agent: Agent, dx: int, dy: int) -> float:
        """Move agent in specified direction"""
        if agent.move_points < 1.0:
            return 0.0

        new_pos = Position(agent.position.x + dx, agent.position.y + dy)

        if not self.game_state.is_valid_position(new_pos):
            return -0.1  # Small penalty for invalid move

        agent.position = new_pos
        agent.move_points -= 1.0
        agent.consume_energy(agent.config.energy_consumption_move)

        # Small reward for successful move
        return 0.0

    def _gather_resource(self, agent: Agent) -> float:
        """Gather resource at agent's position"""
        if agent.type == AgentType.SCOUT:
            return 0.0  # Scouts can't gather

        resource = self.game_state.get_resource_at(agent.position)
        if not resource or not resource.is_active:
            return 0.0

        # Check if agent is already gathering
        if agent.gathering_target == resource:
            agent.gathering_progress += 1
            agent.consume_energy(agent.config.energy_consumption_gather)

            # Check if gathering is complete
            required_time = RESOURCE_CONFIGS[resource.resource_type].gather_difficulty
            if agent.gathering_progress >= required_time:
                gathered = resource.gather()
                if gathered > 0:
                    agent.add_resources(resource.resource_type, gathered)
                    reward = RESOURCE_CONFIGS[resource.resource_type].value * 0.1
                    agent.gathering_target = None
                    agent.gathering_progress = 0
                    return reward

        # Start gathering new resource
        else:
            agent.gathering_target = resource
            agent.gathering_progress = 0
            agent.consume_energy(agent.config.energy_consumption_gather)

        return 0.0

    def _transfer_resources(self, agent: Agent) -> float:
        """Transfer resources between agents"""
        if not agent.is_carrying_resources:
            return 0.0

        # Find adjacent agents
        adjacent_agents = []
        for other_agent in self.agents.values():
            if other_agent.id != agent.id and agent.position.is_adjacent(other_agent.position):
                adjacent_agents.append(other_agent)

        if not adjacent_agents:
            return 0.0

        # Find transporter for workers, or worker for transporters
        target_agent = None
        if agent.type == AgentType.WORKER:
            target_agent = next((a for a in adjacent_agents if a.type == AgentType.TRANSPORTER), None)
        elif agent.type == AgentType.TRANSPORTER:
            target_agent = next((a for a in adjacent_agents if a.type == AgentType.WORKER), None)

        if not target_agent or not target_agent.can_carry_more:
            return 0.0

        # Transfer resources
        transferred_value = 0.0
        for rtype in ResourceType:
            if agent.inventory[rtype] > 0:
                amount = agent.remove_resources(rtype, agent.inventory[rtype])
                added = target_agent.add_resources(rtype, amount)
                transferred_value += added * RESOURCE_CONFIGS[rtype].value * 0.05

        agent.consume_energy(agent.config.energy_consumption_transfer)
        return transferred_value

    def _deposit_resources(self, agent: Agent) -> float:
        """Deposit resources at base"""
        if not agent.is_at_base or not agent.is_carrying_resources:
            return 0.0

        # Deposit all resources
        deposited_resources = agent.clear_inventory()
        self.game_state.deposit_resources(deposited_resources)

        # Calculate reward based on deposited value
        reward = 0.0
        for rtype, amount in deposited_resources.items():
            reward += amount * RESOURCE_CONFIGS[rtype].value * 0.5  # 50% of full value

        return reward

    def _calculate_team_reward(self) -> float:
        """Calculate team-wide reward"""
        reward = 0.0

        # Penalty for time passing
        reward -= 0.01

        # Small bonus for resource diversity
        if self.game_state.deposited_resources[ResourceType.GOLD] > 0:
            reward += 0.1
        if self.game_state.deposited_resources[ResourceType.WOOD] > 0:
            reward += 0.05

        return reward

    def _get_observation(self, agent_id: str) -> np.ndarray:
        """Get observation for a specific agent - optimized version"""
        agent = self.agents[agent_id]
        obs = np.zeros(60, dtype=np.float32)  # Reduced from 80 to 60 dimensions
        idx = 0

        # Agent self state (10 dimensions) - same as before
        obs[idx] = agent.position.x / self.config.grid_size
        idx += 1
        obs[idx] = agent.position.y / self.config.grid_size
        idx += 1

        # Role one-hot encoding (3 dimensions)
        obs[idx + agent.type.value] = 1.0  # Direct indexing instead of list creation
        idx += 3

        # Inventory (2 dimensions)
        obs[idx] = agent.inventory[ResourceType.GOLD] / 10.0
        idx += 1
        obs[idx] = agent.inventory[ResourceType.WOOD] / 10.0
        idx += 1

        # Energy and cooldown (2 dimensions)
        obs[idx] = agent.energy / 100.0
        idx += 1
        obs[idx] = agent.action_cooldown / 2.0
        idx += 1

        # Distance to base (1 dimension)
        base_distance = agent.position.distance_to(self.game_state.base_position)
        obs[idx] = base_distance / (self.config.grid_size * 2)
        idx += 1

        # Time remaining (1 dimension)
        obs[idx] = 1.0 - (self.game_state.current_step / self.game_state.max_steps)
        idx += 1

        # Optimized visible entities detection (40 dimensions total)
        # Use a simpler, faster approach: limit to immediate vicinity
        vision_range = min(agent.config.vision_range, 3)  # Cap vision range for performance
        max_entities = 6  # Reduced from 10 to 6 entities

        # Pre-compute bounds to avoid repeated calculations
        min_x = max(0, agent.position.x - vision_range)
        max_x = min(self.config.grid_size, agent.position.x + vision_range + 1)
        min_y = max(0, agent.position.y - vision_range)
        max_y = min(self.config.grid_size, agent.position.y + vision_range + 1)

        # Quick scan for nearby entities
        entity_count = 0
        for dx in range(-vision_range, vision_range + 1):
            if entity_count >= max_entities:
                break
            for dy in range(-vision_range, vision_range + 1):
                if entity_count >= max_entities:
                    break

                check_x = agent.position.x + dx
                check_y = agent.position.y + dy

                # Skip if out of bounds
                if check_x < min_x or check_x >= max_x or check_y < min_y or check_y >= max_y:
                    continue

                check_pos = Position(check_x, check_y)

                # Check for other agents first (usually more important)
                for other_agent in self.agents.values():
                    if other_agent.id != agent_id and other_agent.position == check_pos:
                        # Relative position (2 dimensions)
                        obs[idx] = dx / vision_range
                        idx += 1
                        obs[idx] = dy / vision_range
                        idx += 1

                        # Agent type (3 dimensions)
                        obs[idx + other_agent.type.value] = 1.0
                        idx += 3

                        entity_count += 1
                        break
                else:
                    # Check for resources if no agent found
                    for resource in self.game_state.resources:
                        if (resource.is_active and resource.position == check_pos):
                            # Relative position (2 dimensions)
                            obs[idx] = dx / vision_range
                            idx += 1
                            obs[idx] = dy / vision_range
                            idx += 1

                            # Resource type (2 dimensions)
                            obs[idx + resource.resource_type.value] = 1.0
                            idx += 2

                            # Resource quantity (1 dimension)
                            obs[idx] = resource.remaining_quantity / 5.0
                            idx += 1

                            entity_count += 1
                            break

        # Pad remaining entity observations (5 dimensions per entity expected)
        expected_dims = 10 + max_entities * 5  # First 10 + entities
        padding = expected_dims - idx
        if padding > 0:
            obs[idx:idx+padding] = 0.0
            idx += padding

        # Simplified messages (10 dimensions) - keep as is
        message_dim = 3
        recent_messages = self.game_state.message_history[-message_dim:]
        for i, msg in enumerate(recent_messages):
            if i < message_dim:
                # Simple message encoding (based on agent type)
                if msg['agent_id'].startswith('scout'):
                    obs[idx] = 0.3
                elif msg['agent_id'].startswith('worker'):
                    obs[idx] = 0.6
                else:
                    obs[idx] = 0.9
                idx += 1

        return obs

    def _get_info(self, agent_id: str) -> Dict[str, Any]:
        """Get info dictionary for a specific agent"""
        agent = self.agents[agent_id]
        return {
            'agent_id': agent_id,
            'position': (agent.position.x, agent.position.y),
            'energy': agent.energy,
            'inventory': dict(agent.inventory),
            'is_at_base': agent.is_at_base,
            'total_score': self.game_state.total_score
        }

    def _render_step(self):
        """Render current step"""
        if self.renderer:
            self.renderer.render(self.game_state)

    def render(self, mode='human'):
        """Render the environment"""
        if self.renderer:
            self.renderer.render(self.game_state, mode=mode)

    def close(self):
        """Close the environment"""
        if self.renderer:
            self.renderer.close()

    def get_avail_actions(self, agent_id: str) -> List[int]:
        """Get available actions for an agent (for action masking)"""
        agent = self.agents[agent_id]
        avail_actions = []

        for action in ActionType:
            if agent.can_perform_action(action):
                avail_actions.append(action.value)

        return avail_actions

    def get_global_state(self) -> np.ndarray:
        """Get global state representation (for CTDE algorithms) - optimized version"""
        # Reduced dimension from 200 to 120 for performance
        global_state = np.zeros(120, dtype=np.float32)
        idx = 0

        # Simplified agent states (6 agents * 12 dimensions = 72)
        for agent_id in sorted(self.agent_ids):
            agent = self.agents[agent_id]

            # Position and role (5 dimensions) - keep as is
            global_state[idx] = agent.position.x / self.config.grid_size
            idx += 1
            global_state[idx] = agent.position.y / self.config.grid_size
            idx += 1
            global_state[idx + agent.type.value] = 1.0  # Direct indexing
            idx += 3

            # Simplified inventory and status (7 dimensions) -> reduced to 7
            global_state[idx] = agent.inventory[ResourceType.GOLD] / 10.0
            idx += 1
            global_state[idx] = agent.inventory[ResourceType.WOOD] / 10.0
            idx += 1
            global_state[idx] = agent.energy / 100.0
            idx += 1
            global_state[idx] = 1.0 if agent.is_at_base else 0.0
            idx += 1
            global_state[idx] = 1.0 if agent.is_carrying_resources else 0.0
            idx += 1
            global_state[idx] = agent.move_points / 2.0
            idx += 1

        # Simplified resource summary (24 dimensions) instead of individual resources
        # Count resources by type and region
        active_gold = sum(1 for r in self.game_state.resources
                         if r.is_active and r.resource_type == ResourceType.GOLD)
        active_wood = sum(1 for r in self.game_state.resources
                         if r.is_active and r.resource_type == ResourceType.WOOD)

        # Resource counts (2 dimensions)
        global_state[idx] = active_gold / 10.0
        idx += 1
        global_state[idx] = active_wood / 20.0
        idx += 1

        # Resource locations by quadrant (4 quadrants * 2 types * 2 avg_pos = 16 dimensions)
        for quadrant in [(0, 1), (0, -1), (1, 0), (-1, 0)]:  # NE, SE, SW, NW
            qx_offset, qy_offset = quadrant
            for rtype in [ResourceType.GOLD, ResourceType.WOOD]:
                resources_in_quad = []
                for r in self.game_state.resources:
                    if (r.is_active and r.resource_type == rtype):
                        center_x = self.config.grid_size / 2
                        center_y = self.config.grid_size / 2
                        if ((r.position.x - center_x) * qx_offset >= 0 and
                            (r.position.y - center_y) * qy_offset >= 0):
                            resources_in_quad.append(r)

                if resources_in_quad:
                    avg_x = sum(r.position.x for r in resources_in_quad) / len(resources_in_quad)
                    avg_y = sum(r.position.y for r in resources_in_quad) / len(resources_in_quad)
                    global_state[idx] = avg_x / self.config.grid_size
                    idx += 1
                    global_state[idx] = avg_y / self.config.grid_size
                    idx += 1
                else:
                    global_state[idx] = 0.0
                    idx += 1
                    global_state[idx] = 0.0
                    idx += 1

        # Global game state (24 dimensions) - simplified
        global_state[idx] = self.game_state.total_score / 1000.0
        idx += 1
        global_state[idx] = self.game_state.deposited_resources[ResourceType.GOLD] / 20.0
        idx += 1
        global_state[idx] = self.game_state.deposited_resources[ResourceType.WOOD] / 50.0
        idx += 1
        global_state[idx] = self.game_state.current_step / self.game_state.max_steps
        idx += 1

        # Team composition summary (3 dimensions)
        agent_types = [agent.type for agent in self.agents.values()]
        global_state[idx] = sum(1 for t in agent_types if t == AgentType.SCOUT) / 6.0
        idx += 1
        global_state[idx] = sum(1 for t in agent_types if t == AgentType.WORKER) / 6.0
        idx += 1
        global_state[idx] = sum(1 for t in agent_types if t == AgentType.TRANSPORTER) / 6.0
        idx += 1

        # Pad remaining dimensions
        global_state[idx:] = 0.0

        return global_state


# Factory function for easy environment creation
def create_hrg_env(difficulty: str = "normal", **kwargs) -> HRGEnv:
    """
    Create HRG environment with predefined difficulty settings

    Args:
        difficulty: Difficulty level ("easy", "normal", "hard", "fast_training")
        **kwargs: Additional configuration parameters

    Returns:
        HRGEnv: Configured environment
    """
    if difficulty == "easy":
        config = HRGConfig(
            grid_size=8,
            max_steps=300,
            num_obstacles=0,
            num_gold=2,
            num_wood=15,
            **kwargs
        )
    elif difficulty == "hard":
        config = HRGConfig(
            grid_size=12,
            max_steps=150,
            num_obstacles=20,
            num_gold=4,
            num_wood=8,
            **kwargs
        )
    elif difficulty == "fast_training":
        config = HRGConfig(
            grid_size=6,
            max_steps=100,
            num_obstacles=0,
            num_gold=1,
            num_wood=5,
            render_mode=None,
            **kwargs
        )
    else:  # normal
        config = HRGConfig(**kwargs)

    return HRGEnv(config)