"""
DEM (Dynamic Escort Mission) Environment

A multi-agent reinforcement learning environment where agents dynamically form roles
to escort a VIP through dangerous territory while dealing with various threats.
"""

import numpy as np
import time
import random
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from gymnasium import spaces

from .core import (
    Position, Agent, VIP, Threat, ActionType, MessageType, ThreatType,
    TerrainType, GameState, DIRECTIONS
)
from .config import DEMConfig

# Configure logging
logger = logging.getLogger(__name__)


class DEMEnv:
    """
    Dynamic Escort Mission Environment

    A multi-agent environment where special forces agents must escort a VIP
    through dangerous territory while dynamically forming roles to deal with
    various threats.
    """

    def __init__(self, config: DEMConfig = None):
        """
        Initialize DEM environment

        Args:
            config: Environment configuration
        """
        self.config = config if config is not None else DEMConfig()

        # Validate configuration
        self.config.validate()

        # Set random seed
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
            random.seed(self.config.seed)

        # Initialize game state
        self.game_state = GameState(grid_size=self.config.grid_size)
        self._apply_config_to_state()

        # Initialize observation and action spaces
        self._init_spaces()

        # Episode tracking
        self.episode_start_time = None
        self.total_episodes = 0

        # Statistics tracking
        self.episode_stats = []

        # VIP pathfinding tracking
        self.vip_position_history = []  # Track recent VIP positions
        self.vip_stuck_threshold = 6    # Steps to consider VIP stuck
        self.vip_last_progress_step = 0  # Last step VIP made progress

        # Initialize renderer
        self.renderer = None
        self._setup_renderer()

        logger.info(f"DEM Environment initialized with {self.config.num_agents} agents "
                   f"on {self.config.grid_size}x{self.config.grid_size} grid")

    def _apply_config_to_state(self) -> None:
        """Apply configuration to game state"""
        # Update game state limits
        self.game_state.max_steps = self.config.max_steps

        # Apply terrain from config
        self._apply_terrain_config()

        # Update VIP configuration
        self.game_state.vip.pos = Position(*self.config.vip_initial_pos)
        self.game_state.vip.target_pos = Position(*self.config.vip_target_pos)
        self.game_state.vip.hp = self.config.vip_hp
        self.game_state.vip.max_hp = self.config.vip_hp
        self.game_state.vip.vision_range = self.config.vip_vision_range
        self.game_state.vip.max_move_cooldown = self.config.vip_move_cooldown

        # Update agent configurations
        self._apply_agent_config()

        # Clear initial threats (will spawn according to timer)
        self.game_state.threats.clear()

    def _apply_terrain_config(self) -> None:
        """Apply terrain configuration from config"""
        # Start with open terrain
        terrain = np.full((self.config.grid_size, self.config.grid_size),
                         TerrainType.OPEN, dtype=object)

        # Apply rivers
        for x, y in self.config.river_positions:
            if 0 <= x < self.config.grid_size and 0 <= y < self.config.grid_size:
                terrain[x, y] = TerrainType.RIVER

        # Apply forests
        for x, y in self.config.forest_positions:
            if 0 <= x < self.config.grid_size and 0 <= y < self.config.grid_size:
                if terrain[x, y] != TerrainType.RIVER:  # Don't overwrite rivers
                    terrain[x, y] = TerrainType.FOREST

        self.game_state.terrain = terrain

    def _apply_agent_config(self) -> None:
        """Apply agent configuration"""
        # Clear existing agents
        self.game_state.agents.clear()

        # Create agents with configured positions
        for i in range(min(self.config.num_agents, len(self.config.agent_initial_positions))):
            agent_id = f"agent_{i}"
            pos = Position(*self.config.agent_initial_positions[i])

            agent = Agent(agent_id, pos)
            agent.hp = self.config.agent_hp
            agent.max_hp = self.config.agent_hp
            agent.damage = self.config.agent_damage
            agent.range = self.config.agent_range
            agent.max_attack_cooldown = self.config.agent_attack_cooldown

            self.game_state.agents[agent_id] = agent

    def _init_spaces(self) -> None:
        """Initialize observation and action spaces"""
        # Each agent has the same action space (10 discrete actions)
        self.action_space = spaces.Discrete(10)

        # Calculate observation space dimensions
        obs_dim = self._calculate_observation_dim()
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Global state space for centralized training
        self.global_state_dim = self._calculate_global_state_dim()

    def _calculate_observation_dim(self) -> int:
        """Calculate the dimension of observation space"""
        # Self state: 8 dimensions
        # pos (2 normalized), hp (1), cooldown (1), guarding (1), vip_dist (1), target_dist (1), in_forest (1)
        self_dim = 8

        # VIP state: 6 dimensions
        # visible (1), hp (1), relative_pos (2), under_attack (1), adjacent (1)
        vip_dim = 6

        # Teammates: max 2 teammates * 6 dimensions each
        # relative_pos (2), hp (1), adjacent_to_vip (1), guarding (1), cooldown (1)
        teammates_dim = 12

        # Threats: max 5 threats * 5 dimensions each
        # type (1), relative_pos (2), hp (1), cooldown (1)
        threats_dim = 25

        # Communication: 3 messages * 2 dimensions each
        # type (1), age (1)
        comm_dim = 6

        # Additional info: 2 dimensions
        # step_normalized (1), constant (1)
        info_dim = 2

        total_dim = self_dim + vip_dim + teammates_dim + threats_dim + comm_dim + info_dim
        return total_dim

    def _calculate_global_state_dim(self) -> int:
        """Calculate global state dimension"""
        # VIP: 4 dimensions
        vip_dim = 4

        # Agents: 3 agents * 4 dimensions each
        agents_dim = 12

        # Threats: 5 threats * 4 dimensions each
        threats_dim = 20

        # Statistics: 5 dimensions
        stats_dim = 5

        return vip_dim + agents_dim + threats_dim + stats_dim

    def reset(self) -> Dict[str, np.ndarray]:
        """Reset environment for new episode"""
        # Reset game state
        self.game_state = GameState(grid_size=self.config.grid_size)
        self._apply_config_to_state()

        # Reset episode tracking
        self.episode_start_time = time.time()
        self.total_episodes += 1

        # Reset VIP pathfinding tracking
        self.vip_position_history = []
        self.vip_last_progress_step = 0

        # Reset statistics
        self.game_state.stats = {
            'vip_damage_taken': 0,
            'threats_killed': 0,
            'agents_killed': 0,
            'messages_sent': 0,
            'body_blocks': 0,
            'long_range_kills': 0,
            'vip_distance_to_target': 0,
            'agents_adjacent_to_vip': 0,
            'agents_ahead_of_vip': 0,
            'agent_spread': 0,
        }

        logger.info(f"Episode {self.total_episodes} started")

        return self.get_observations()

    def step(self, actions: Dict[str, int]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, Any]]:
        """
        Execute one step in the environment

        Args:
            actions: Dictionary mapping agent IDs to actions

        Returns:
            Tuple of (observations, rewards, dones, info)
        """
        if self.game_state.is_terminated:
            observations = self.get_observations()
            rewards = self._get_zero_rewards()
            dones = {agent_id: True for agent_id in self.game_state.agents.keys()}
            return observations, rewards, dones, {}

        # Store previous state for reward calculation
        prev_stats = self.game_state.stats.copy()

        # Execute agent actions
        self._execute_agent_actions(actions)

        # Execute VIP movement
        self._execute_vip_movement()

        # Spawn threats if needed
        self._spawn_threats()

        # Execute threat actions
        self._execute_threat_actions()

        # Update cooldowns
        self._update_cooldowns()

        # Update game statistics
        self._update_game_statistics()

        # Check termination conditions
        terminated = self._check_termination()

        # Calculate rewards
        rewards = self._calculate_rewards(prev_stats, terminated)

        # Update step counter
        self.game_state.current_step += 1

        # Get observations
        observations = self.get_observations()

        # Create dones dictionary - all agents have same episode status
        dones = {agent_id: terminated for agent_id in self.game_state.agents.keys()}

        # Create info dictionary
        info = self._get_info(terminated)

        if terminated:
            self._log_episode_stats()

        # Handle rendering
        if self.renderer is not None:
            self._render_step()

        return observations, rewards, dones, info

    def _execute_agent_actions(self, actions: Dict[str, int]) -> None:
        """Execute actions for all agents"""
        for agent_id, action in actions.items():
            if agent_id not in self.game_state.agents:
                continue

            agent = self.game_state.agents[agent_id]
            if not agent.is_alive:
                continue

            try:
                action_type = ActionType(action)
            except ValueError:
                logger.warning(f"Invalid action {action} for agent {agent_id}")
                continue

            self._execute_single_agent_action(agent, action_type)

    def _execute_single_agent_action(self, agent: Agent, action: ActionType) -> None:
        """Execute action for a single agent"""
        if action == ActionType.STAY:
            return

        elif action in [ActionType.MOVE_UP, ActionType.MOVE_DOWN,
                       ActionType.MOVE_LEFT, ActionType.MOVE_RIGHT]:
            self._execute_agent_move(agent, action)

        elif action == ActionType.ATTACK:
            self._execute_agent_attack(agent)

        elif action == ActionType.GUARD_VIP:
            self._execute_agent_guard(agent)

        elif action == ActionType.WARN_THREAT:
            self._execute_agent_warn(agent, MessageType.THREAT_WARNING)

        elif action == ActionType.ALL_CLEAR:
            self._execute_agent_warn(agent, MessageType.ALL_CLEAR)

        elif action == ActionType.OBSERVE:
            # Observation action - just gather information
            pass

    def _execute_agent_move(self, agent: Agent, action: ActionType) -> None:
        """Execute movement action for agent"""
        direction = DIRECTIONS[action]
        new_pos = agent.pos + direction

        # Check if move is valid
        if not self.game_state.is_valid_position(new_pos):
            return

        if self.game_state.is_position_occupied(new_pos, exclude_id=agent.id):
            return

        # Execute move
        agent.pos = new_pos
        agent.is_guarding = False  # Moving cancels guard state

    def _execute_agent_attack(self, agent: Agent) -> None:
        """Execute attack action for agent"""
        if not agent.can_attack():
            return

        # Find threats in range
        targets = []
        for threat in self.game_state.threats.values():
            if threat.is_alive and agent.pos.manhattan_distance(threat.pos) <= agent.range:
                targets.append(threat)

        if not targets:
            return

        # Attack closest threat
        targets.sort(key=lambda t: agent.pos.manhattan_distance(t.pos))
        target = targets[0]

        # Calculate damage
        terrain = self.game_state.terrain[target.pos.x, target.pos.y]
        damage_reduction = 1.0
        if terrain == TerrainType.FOREST:
            damage_reduction = self.config.forest_damage_reduction

        damage = int(agent.damage * damage_reduction)

        # Apply damage
        target.take_damage(damage)
        agent.attack()

        # Check if threat was killed
        if not target.is_alive:
            self.game_state.stats['threats_killed'] += 1

            # Check for long range kill
            distance = agent.pos.manhattan_distance(target.pos)
            if distance >= 6:
                self.game_state.stats['long_range_kills'] += 1

    def _execute_agent_guard(self, agent: Agent) -> None:
        """Execute guard action for agent"""
        # Can only guard if adjacent to VIP
        if agent.pos.manhattan_distance(self.game_state.vip.pos) == 1:
            agent.is_guarding = True
        else:
            agent.is_guarding = False

    def _execute_agent_warn(self, agent: Agent, message_type: MessageType) -> None:
        """Execute communication action for agent"""
        self.game_state.add_message(agent.id, message_type)

    def _execute_vip_movement(self) -> None:
        """Execute VIP autonomous movement with intelligent pathfinding"""
        if not self.game_state.vip.can_move():
            self.game_state.vip.update_cooldown()
            return

        vip_pos = self.game_state.vip.pos
        target_pos = self.game_state.vip.target_pos

        # Check if VIP has reached target
        if vip_pos == target_pos:
            logger.debug(f"VIP has reached target at {target_pos}")
            return

        # Find nearby threats
        nearby_threats = []
        for threat in self.game_state.threats.values():
            if threat.is_alive and vip_pos.manhattan_distance(threat.pos) <= 3:
                nearby_threats.append(threat)

        # Detect if VIP is stuck/lingering
        is_stuck = self._is_vip_stuck(vip_pos, target_pos)

        if nearby_threats:
            # Use evasive pathfinding when threats are nearby
            best_pos = self._find_evasive_position(vip_pos, nearby_threats)
        elif is_stuck:
            # Use wall-following strategy when stuck
            logger.debug(f"VIP is stuck at {vip_pos}, using wall-following strategy")
            best_pos = self._follow_obstacle_edge(vip_pos, target_pos)
        else:
            # Use intelligent pathfinding towards target
            best_pos = self._find_path_to_target(vip_pos, target_pos)

        # Execute move if a valid position was found
        if best_pos and best_pos != vip_pos:
            self.game_state.vip.pos = best_pos
            # Update position history for stuck detection
            self._update_vip_position_history(best_pos)
            logger.debug(f"VIP moved from {vip_pos} to {best_pos}")
        elif best_pos == vip_pos:
            # VIP didn't move, update stagnation counter
            self._update_vip_position_history(vip_pos)

        self.game_state.vip.move()
        self.game_state.vip.update_cooldown()

    def _update_vip_position_history(self, pos: Position) -> None:
        """Update VIP position history for stuck detection"""
        self.vip_position_history.append(pos)

        # Keep only recent positions
        if len(self.vip_position_history) > self.vip_stuck_threshold:
            self.vip_position_history.pop(0)

        # Check if VIP made progress (got closer to target)
        target_pos = self.game_state.vip.target_pos
        current_distance = pos.manhattan_distance(target_pos)

        # Update last progress step if VIP got closer to target
        if len(self.vip_position_history) > 1:
            prev_distance = self.vip_position_history[-2].manhattan_distance(target_pos)
            if current_distance < prev_distance:
                self.vip_last_progress_step = self.game_state.current_step

    def _is_vip_stuck(self, current_pos: Position, target_pos: Position) -> bool:
        """Detect if VIP is stuck or lingering in one area"""
        # Not enough history yet
        if len(self.vip_position_history) < 4:
            return False

        # Check if VIP hasn't made progress for many steps
        steps_since_progress = self.game_state.current_step - self.vip_last_progress_step
        if steps_since_progress > 8:  # No progress for 8+ steps
            return True

        # Check if VIP is visiting the same positions repeatedly
        unique_positions = len(set(self.vip_position_history[-6:]))  # Last 6 positions
        if unique_positions <= 3:  # Only 3 or fewer unique positions
            return True

        # Check if VIP is staying in the same area (small bounding box)
        recent_positions = self.vip_position_history[-4:]
        x_coords = [pos.x for pos in recent_positions]
        y_coords = [pos.y for pos in recent_positions]

        if max(x_coords) - min(x_coords) <= 1 and max(y_coords) - min(y_coords) <= 1:
            # VIP is moving within a 2x2 area
            return True

        return False

    def _follow_obstacle_edge(self, current_pos: Position, target_pos: Position) -> Optional[Position]:
        """Follow obstacle edge to get around obstacles when stuck"""
        logger.debug(f"VIP using wall-following strategy from {current_pos} to {target_pos}")

        # Find the direction to target
        to_target = Position(
            target_pos.x - current_pos.x,
            target_pos.y - current_pos.y
        )

        # Normalize direction to primary movement
        primary_dir = None
        if abs(to_target.x) >= abs(to_target.y):
            primary_dir = 'horizontal' if to_target.x != 0 else 'vertical'
        else:
            primary_dir = 'vertical' if to_target.y != 0 else 'horizontal'

        # Try to find a path along obstacle edges
        best_pos = self._find_edge_path(current_pos, target_pos, primary_dir)

        if best_pos and best_pos != current_pos:
            logger.debug(f"VIP edge-following found path: {current_pos} -> {best_pos}")
            return best_pos

        # Fallback: try to move in target direction with obstacle avoidance
        return self._greedy_move_with_obstacle_avoidance(current_pos, target_pos)

    def _find_edge_path(self, current_pos: Position, target_pos: Position, primary_dir: str) -> Optional[Position]:
        """Find path along obstacle edges"""
        # Define movement directions based on primary direction
        if primary_dir == 'horizontal':
            # Moving primarily horizontal, try vertical movements first
            edge_directions = [
                Position(0, 1), Position(0, -1),  # Vertical
                Position(1, 0), Position(-1, 0),   # Horizontal
                Position(1, 1), Position(-1, -1),  # Diagonals
                Position(1, -1), Position(-1, 1)
            ]
        else:
            # Moving primarily vertical, try horizontal movements first
            edge_directions = [
                Position(1, 0), Position(-1, 0),   # Horizontal
                Position(0, 1), Position(0, -1),   # Vertical
                Position(1, 1), Position(-1, -1),  # Diagonals
                Position(1, -1), Position(-1, 1)
            ]

        best_pos = None
        best_score = float('-inf')

        for direction in edge_directions:
            new_pos = current_pos + direction

            if not self._is_vip_valid_position(new_pos):
                continue

            # Calculate score based on multiple factors
            score = self._calculate_edge_score(new_pos, target_pos, current_pos)

            if score > best_score:
                best_score = score
                best_pos = new_pos

        return best_pos if best_pos is not None else current_pos

    def _calculate_edge_score(self, pos: Position, target_pos: Position, current_pos: Position) -> float:
        """Calculate score for edge-following positions"""
        score = 0.0

        # Distance to target (primary factor)
        target_distance = pos.manhattan_distance(target_pos)
        current_target_distance = current_pos.manhattan_distance(target_pos)

        if target_distance < current_target_distance:
            score += 2.0  # Strong bonus for getting closer
        else:
            score -= 0.5  # Small penalty for moving away

        # Bonus for positions near obstacles (edge following)
        if self._is_near_obstacle(pos):
            score += 0.8

        # Penalty for staying too close to current position
        if pos.manhattan_distance(current_pos) == 0:
            score -= 2.0

        # Bonus for maintaining momentum (prefer movement in consistent direction)
        if len(self.vip_position_history) >= 2:
            prev_pos = self.vip_position_history[-2]
            last_move = current_pos - prev_pos
            this_move = pos - current_pos

            # Reward consistent movement direction
            if (last_move.x * this_move.x >= 0) and (last_move.y * this_move.y >= 0):
                score += 0.3

        return score

    def _is_near_obstacle(self, pos: Position) -> bool:
        """Check if position is near an obstacle (river)"""
        directions = [
            Position(0, 1), Position(0, -1), Position(1, 0), Position(-1, 0),
            Position(1, 1), Position(1, -1), Position(-1, 1), Position(-1, -1)
        ]

        for direction in directions:
            neighbor = pos + direction
            if (0 <= neighbor.x < self.config.grid_size and
                0 <= neighbor.y < self.config.grid_size):
                terrain = self.game_state.terrain[neighbor.x][neighbor.y]
                if terrain == TerrainType.RIVER:
                    return True

        return False

    def _greedy_move_with_obstacle_avoidance(self, current_pos: Position, target_pos: Position) -> Optional[Position]:
        """Greedy movement with enhanced obstacle avoidance"""
        directions = [
            Position(0, 1), Position(0, -1), Position(1, 0), Position(-1, 0),
            Position(1, 1), Position(1, -1), Position(-1, 1), Position(-1, -1)
        ]

        best_pos = None
        best_score = float('-inf')

        for direction in directions:
            new_pos = current_pos + direction

            if not self._is_vip_valid_position(new_pos):
                continue

            # Calculate composite score
            target_distance = new_pos.manhattan_distance(target_pos)
            obstacle_penalty = 0.0

            # Check if this move gets us closer to obstacles (avoid if possible)
            if self._is_near_obstacle(new_pos):
                obstacle_penalty = 0.2  # Small penalty for being near obstacles

            # Score: prioritize getting to target, avoid obstacles
            score = -target_distance - obstacle_penalty

            if score > best_score:
                best_score = score
                best_pos = new_pos

        return best_pos if best_pos is not None else current_pos

    def _find_path_to_target(self, current_pos: Position, target_pos: Position) -> Optional[Position]:
        """Find the best next position towards target using intelligent pathfinding"""
        # Use A* pathfinding if distance is reasonable
        distance = current_pos.manhattan_distance(target_pos)
        if distance <= 8:  # Use A* for short to medium distances
            return self._astar_next_step(current_pos, target_pos)
        else:
            # Use greedy approach for long distances
            return self._greedy_move_towards_target(current_pos, target_pos)

    def _find_evasive_position(self, current_pos: Position, threats: List) -> Optional[Position]:
        """Find the best position to evade nearby threats"""
        best_pos = None
        best_score = float('-inf')

        # Check all 8 directions
        directions = [
            Position(0, 1), Position(0, -1), Position(1, 0), Position(-1, 0),
            Position(1, 1), Position(1, -1), Position(-1, 1), Position(-1, -1)
        ]

        for direction in directions:
            new_pos = current_pos + direction

            # Check if position is valid
            if not self._is_vip_valid_position(new_pos):
                continue

            # Calculate score based on distance from threats and progress to target
            score = self._calculate_evasion_score(new_pos, threats)

            # Add bias towards target
            to_target = self.game_state.vip.target_pos - new_pos
            target_distance = new_pos.manhattan_distance(self.game_state.vip.target_pos)
            current_target_distance = current_pos.manhattan_distance(self.game_state.vip.target_pos)

            if target_distance < current_target_distance:
                score += 0.3  # Bonus for moving towards target

            if score > best_score:
                best_score = score
                best_pos = new_pos

        return best_pos if best_pos is not None else current_pos

    def _calculate_evasion_score(self, pos: Position, threats: List) -> float:
        """Calculate evasion score for a position"""
        score = 0.0

        for threat in threats:
            distance = pos.manhattan_distance(threat.pos)
            if distance <= threat.range:
                # Penalty for being in attack range
                score -= 3.0 / (distance + 0.1)
            else:
                # Reward for being far from threat
                score += distance * 0.5

        return score

    def _is_vip_valid_position(self, pos: Position) -> bool:
        """Check if position is valid for VIP movement"""
        return (self.game_state.is_valid_position(pos) and
                not self.game_state.is_position_occupied(pos, exclude_id="vip"))

    def _greedy_move_towards_target(self, current_pos: Position, target_pos: Position) -> Optional[Position]:
        """Greedy approach to move towards target"""
        directions = [
            Position(0, 1), Position(0, -1), Position(1, 0), Position(-1, 0),
            Position(1, 1), Position(1, -1), Position(-1, 1), Position(-1, -1)
        ]

        best_pos = None
        best_distance = float('inf')

        for direction in directions:
            new_pos = current_pos + direction

            if not self._is_vip_valid_position(new_pos):
                continue

            distance = new_pos.manhattan_distance(target_pos)
            if distance < best_distance:
                best_distance = distance
                best_pos = new_pos

        return best_pos if best_pos is not None else current_pos

    def _astar_next_step(self, current_pos: Position, target_pos: Position) -> Optional[Position]:
        """A* pathfinding to find next step towards target"""
        if current_pos == target_pos:
            return current_pos

        # Simple A* implementation for immediate next step
        open_set = [(current_pos, 0, current_pos.manhattan_distance(target_pos))]
        closed_set = set()
        came_from = {}

        while open_set:
            current, g_score, f_score = min(open_set, key=lambda x: x[2])
            open_set.remove((current, g_score, f_score))

            if current in closed_set:
                continue

            closed_set.add(current)

            if current == target_pos:
                # Reconstruct path and return first step
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                if len(path) > 0:
                    return path[-1]
                break

            # Check neighbors
            directions = [
                Position(0, 1), Position(0, -1), Position(1, 0), Position(-1, 0),
                Position(1, 1), Position(1, -1), Position(-1, 1), Position(-1, -1)
            ]

            for direction in directions:
                neighbor = current + direction

                if not self._is_vip_valid_position(neighbor):
                    continue

                if neighbor in closed_set:
                    continue

                tentative_g = g_score + 1
                tentative_f = tentative_g + neighbor.manhattan_distance(target_pos)

                # Check if neighbor is already in open set with better score
                existing = None
                for item in open_set:
                    if item[0] == neighbor:
                        existing = item
                        break

                if existing is None or tentative_g < existing[1]:
                    came_from[neighbor] = current
                    if existing is not None:
                        open_set.remove(existing)
                    open_set.append((neighbor, tentative_g, tentative_f))

        # If no path found, fall back to greedy approach
        return self._greedy_move_towards_target(current_pos, target_pos)

    def _spawn_threats(self) -> None:
        """Spawn new threats according to schedule"""
        if self.game_state.current_step < self.config.threat_spawn_initial_delay:
            return

        self.game_state.update_threat_spawn_interval()

        if self.game_state.threat_spawn_timer <= 0:
            self.game_state.spawn_threat()
            self.game_state.threat_spawn_timer = self.game_state.threat_spawn_interval
        else:
            self.game_state.threat_spawn_timer -= 1

    def _execute_threat_actions(self) -> None:
        """Execute actions for all threats"""
        for threat in self.game_state.threats.values():
            if not threat.is_alive:
                continue

            # Rushers move towards VIP
            if threat.type == ThreatType.RUSHER:
                self._execute_rusher_action(threat)
            # Shooters attack from range
            elif threat.type == ThreatType.SHOOTER:
                self._execute_shooter_action(threat)

    def _execute_rusher_action(self, threat: Threat) -> None:
        """Execute action for rusher threat"""
        # Try to move towards VIP
        vip_pos = self.game_state.vip.pos
        threat_pos = threat.pos

        # Calculate movement direction
        to_vip = vip_pos - threat_pos
        move_distance = min(threat.move_range, 1)  # Rushers move 1 step

        if move_distance > 0:
            # Normalize to primary direction
            if abs(to_vip.x) > abs(to_vip.y):
                move_dir = Position(np.sign(to_vip.x), 0)
            else:
                move_dir = Position(0, np.sign(to_vip.y))

            new_pos = threat_pos + move_dir
            if (self.game_state.is_valid_position(new_pos) and
                not self.game_state.is_position_occupied(new_pos, exclude_id=threat.id)):
                threat.pos = new_pos

        # Try to attack VIP if in range
        if threat.can_attack():
            if threat_pos.manhattan_distance(vip_pos) <= threat.range:
                # Calculate damage
                terrain = self.game_state.terrain[vip_pos.x, vip_pos.y]
                damage_reduction = 1.0
                if terrain == TerrainType.FOREST:
                    damage_reduction = self.config.forest_damage_reduction

                # Check for body blocking by guards
                body_block_reduction = 1.0
                num_adjacent_agents = 0
                for agent in self.game_state.agents.values():
                    if agent.is_alive and agent.pos.manhattan_distance(vip_pos) == 1:
                        num_adjacent_agents += 1
                        if agent.is_guarding:
                            body_block_reduction = 0.5
                            self.game_state.stats['body_blocks'] += 1

                total_reduction = damage_reduction * body_block_reduction
                damage = int(threat.damage * total_reduction)

                old_hp = self.game_state.vip.hp
                self.game_state.vip.take_damage(damage, total_reduction)
                new_hp = self.game_state.vip.hp

                self.game_state.stats['vip_damage_taken'] += (old_hp - new_hp)
                self.game_state.vip.is_under_attack = True
                threat.attack()

    def _execute_shooter_action(self, threat: Threat) -> None:
        """Execute action for shooter threat"""
        vip_pos = self.game_state.vip.pos
        threat_pos = threat.pos

        # Shooters only attack, don't move
        if threat.can_attack():
            if threat_pos.manhattan_distance(vip_pos) <= threat.range:
                # Calculate damage
                terrain = self.game_state.terrain[vip_pos.x, vip_pos.y]
                damage_reduction = 1.0
                if terrain == TerrainType.FOREST:
                    damage_reduction = self.config.forest_damage_reduction

                # Check for body blocking by guards
                body_block_reduction = 1.0
                for agent in self.game_state.agents.values():
                    if agent.is_alive and agent.pos.manhattan_distance(vip_pos) == 1:
                        if agent.is_guarding:
                            body_block_reduction = 0.5
                            self.game_state.stats['body_blocks'] += 1

                total_reduction = damage_reduction * body_block_reduction
                damage = int(threat.damage * total_reduction)

                old_hp = self.game_state.vip.hp
                self.game_state.vip.take_damage(damage, total_reduction)
                new_hp = self.game_state.vip.hp

                self.game_state.stats['vip_damage_taken'] += (old_hp - new_hp)
                self.game_state.vip.is_under_attack = True
                threat.attack()

    def _update_cooldowns(self) -> None:
        """Update cooldowns for all entities"""
        # Update agent cooldowns
        for agent in self.game_state.agents.values():
            agent.update_cooldown()

        # Update VIP cooldown
        self.game_state.vip.update_cooldown()

        # Update threat cooldowns
        for threat in self.game_state.threats.values():
            threat.update_cooldown()

        # Reset VIP attack status
        self.game_state.vip.is_under_attack = False

    def _update_game_statistics(self) -> None:
        """Update game statistics"""
        self.game_state.update_statistics()

    def _check_termination(self) -> bool:
        """Check if episode should terminate"""
        if self.game_state.is_terminated:
            return True

        # Check VIP reached target
        if self.game_state.vip.pos.manhattan_distance(self.game_state.vip.target_pos) <= 1:
            self.game_state.is_terminated = True
            self.game_state.termination_reason = "vip_reached_target"
            self.game_state.total_reward += self.config.reward_vip_reach_target
            return True

        # Check VIP death
        if not self.game_state.vip.is_alive:
            self.game_state.is_terminated = True
            self.game_state.termination_reason = "vip_died"
            self.game_state.total_reward += self.config.reward_vip_death
            return True

        # Check step limit
        if self.game_state.current_step >= self.game_state.max_steps:
            self.game_state.is_terminated = True
            self.game_state.termination_reason = "max_steps_reached"
            return True

        # Check time limit
        if self.episode_start_time:
            elapsed_time = time.time() - self.episode_start_time
            if elapsed_time > self.config.episode_time_limit:
                self.game_state.is_terminated = True
                self.game_state.termination_reason = "time_limit_reached"
                return True

        return False

    def _calculate_rewards(self, prev_stats: Dict[str, float], terminated: bool) -> Dict[str, float]:
        """Calculate rewards for all agents"""
        rewards = {agent_id: 0.0 for agent_id in self.game_state.agents.keys()}

        # Team-based rewards (shared among all agents)
        team_reward = 0.0

        # Progress reward
        current_distance = self.game_state.stats['vip_distance_to_target']
        prev_distance = prev_stats.get('vip_distance_to_target', current_distance)
        progress = prev_distance - current_distance
        team_reward += progress * self.config.reward_vip_progress

        # New threats killed
        threats_killed = self.game_state.stats['threats_killed'] - prev_stats.get('threats_killed', 0)
        team_reward += threats_killed * self.config.reward_threat_killed

        # VIP damage penalty
        vip_damage = self.game_state.stats['vip_damage_taken'] - prev_stats.get('vip_damage_taken', 0)
        team_reward += vip_damage * self.config.reward_vip_damage

        # Agent deaths
        agents_killed = self.game_state.stats['agents_killed'] - prev_stats.get('agents_killed', 0)
        team_reward += agents_killed * self.config.reward_agent_death

        # Role emergence rewards
        team_reward += self._calculate_role_emergence_rewards(prev_stats)

        # Communication cost
        messages_sent = self.game_state.stats['messages_sent'] - prev_stats.get('messages_sent', 0)
        team_reward -= messages_sent * self.config.message_cost

        # Distribute team reward equally among alive agents
        alive_agents = [agent_id for agent_id, agent in self.game_state.agents.items()
                        if agent.is_alive]

        if alive_agents:
            reward_per_agent = team_reward / len(alive_agents)
            for agent_id in alive_agents:
                rewards[agent_id] = reward_per_agent

        return rewards

    def _calculate_role_emergence_rewards(self, prev_stats: Dict[str, float]) -> float:
        """Calculate role emergence rewards"""
        role_reward = 0.0

        # Guard reward/penalty
        current_adjacent = self.game_state.stats['agents_adjacent_to_vip']
        if current_adjacent >= 1:
            role_reward += self.config.reward_guard_adjacent
        else:
            role_reward += self.config.reward_guard_missing_penalty

        # Vanguard reward/penalty
        current_ahead = self.game_state.stats['agents_ahead_of_vip']
        if current_ahead >= 1:
            role_reward += self.config.reward_vanguard_ahead
        else:
            role_reward += self.config.reward_vanguard_missing_penalty

        # Spread reward/penalty
        current_spread = self.game_state.stats['agent_spread']
        if 2 <= current_spread <= 5:
            role_reward += self.config.reward_spread_good
        else:
            role_reward += self.config.reward_spread_bad

        # Body block reward
        body_blocks = self.game_state.stats['body_blocks'] - prev_stats.get('body_blocks', 0)
        role_reward += body_blocks * self.config.reward_body_block

        # Long range kill reward
        long_range_kills = self.game_state.stats['long_range_kills'] - prev_stats.get('long_range_kills', 0)
        role_reward += long_range_kills * self.config.reward_long_range_kill

        return role_reward

    def _get_zero_rewards(self) -> Dict[str, float]:
        """Get zero rewards for all agents"""
        return {agent_id: 0.0 for agent_id in self.game_state.agents.keys()}

    def _get_info(self, terminated: bool) -> Dict[str, Any]:
        """Get info dictionary"""
        info = {
            'episode_step': self.game_state.current_step,
            'max_steps': self.game_state.max_steps,
            'vip_hp': self.game_state.vip.hp,
            'vip_max_hp': self.game_state.vip.max_hp,
            'vip_distance_to_target': self.game_state.stats['vip_distance_to_target'],
            'threats_alive': len([t for t in self.game_state.threats.values() if t.is_alive]),
            'agents_alive': len([a for a in self.game_state.agents.values() if a.is_alive]),
            'messages_sent': self.game_state.stats['messages_sent'],
            'threats_killed': self.game_state.stats['threats_killed'],
            'body_blocks': self.game_state.stats['body_blocks'],
            'long_range_kills': self.game_state.stats['long_range_kills'],
        }

        if terminated:
            info['termination_reason'] = self.game_state.termination_reason
            info['total_reward'] = self.game_state.total_reward

        return info

    def get_observations(self) -> Dict[str, np.ndarray]:
        """Get observations for all agents"""
        observations = {}

        for agent_id in self.game_state.agents.keys():
            obs = self.game_state.get_observation_for_agent(agent_id)
            observations[agent_id] = self._encode_observation(obs)

        return observations

    def _encode_observation(self, obs: Dict[str, Any]) -> np.ndarray:
        """Encode observation into numpy array"""
        features = []

        # Self state
        self_state = obs['self']
        features.extend([
            self_state['pos'][0] / self.config.grid_size,
            self_state['pos'][1] / self.config.grid_size,
            self_state['hp'],
            self_state['attack_cooldown'],
            self_state['is_guarding'],
            self_state['vip_distance'],
            self_state['target_distance'],
            self_state['is_in_forest']
        ])

        # VIP state
        vip_state = obs['vip']
        features.extend([
            vip_state['visible'],
            vip_state['hp'] if vip_state['visible'] else 0.0,
            vip_state['relative_pos'][0] / self.config.grid_size if vip_state['visible'] else 0.0,
            vip_state['relative_pos'][1] / self.config.grid_size if vip_state['visible'] else 0.0,
            vip_state['is_under_attack'] if vip_state['visible'] else 0.0,
            vip_state['is_adjacent'] if vip_state['visible'] else 0.0
        ])

        # Teammates (pad to max 2)
        teammates = obs['teammates'][:2]
        while len(teammates) < 2:
            teammates.append({
                'relative_pos': (0, 0),
                'hp': 0.0,
                'is_adjacent_to_vip': 0,
                'is_guarding': 0,
                'attack_cooldown': 0.0
            })

        for teammate in teammates:
            features.extend([
                teammate['relative_pos'][0] / self.config.grid_size,
                teammate['relative_pos'][1] / self.config.grid_size,
                teammate['hp'],
                teammate['is_adjacent_to_vip'],
                teammate['is_guarding'],
                teammate['attack_cooldown']
            ])

        # Threats (pad to max 5)
        threats = obs['threats'][:5]
        while len(threats) < 5:
            threats.append({
                'type': 'rusher',
                'relative_pos': (0, 0),
                'hp': 0.0,
                'attack_cooldown': 0.0
            })

        for threat in threats:
            features.extend([
                1.0 if threat['type'] == 'rusher' else 0.0,
                threat['relative_pos'][0] / self.config.grid_size,
                threat['relative_pos'][1] / self.config.grid_size,
                threat['hp'],
                threat['attack_cooldown']
            ])

        # Communication (pad to max 3)
        messages = obs['communication'][:3]
        while len(messages) < 3:
            messages.append({'type': 1, 'age': 100})

        for message in messages:
            features.extend([
                1.0 if message['type'] == MessageType.THREAT_WARNING else 0.0,
                min(message['age'] / 20.0, 1.0)  # Normalize age
            ])

        # Additional info
        features.extend([
            obs['step'] / obs['max_steps'],
            1.0  # Constant for simplicity
        ])

        return np.array(features, dtype=np.float32)

    def get_global_state(self) -> np.ndarray:
        """Get global state for centralized training"""
        return self.game_state.get_global_state()

    def _setup_renderer(self):
        """Setup renderer based on render mode"""
        if self.config.render_mode in ["human", "rgb_array"]:
            try:
                from .renderer import DEMRenderer
                self.renderer = DEMRenderer(
                    grid_size=self.config.grid_size,
                    cell_size=self.config.render_grid_size,
                    fps=self.config.render_fps
                )
                logger.info(f"Renderer initialized with mode: {self.config.render_mode}")
            except ImportError as e:
                logger.warning(f"Failed to initialize renderer: {e}")
                self.renderer = None

    def _render_step(self):
        """Render current step"""
        if self.renderer:
            self.renderer.render(self.game_state, mode=self.config.render_mode)

    def render(self, mode='human'):
        """Render the environment"""
        if self.renderer:
            return self.renderer.render(self.game_state, mode=mode)
        return None

    def get_env_info(self) -> Dict[str, Any]:
        """Get environment information"""
        return {
            'n_agents': self.config.num_agents,
            'agent_ids': list(self.game_state.agents.keys()),
            'action_space': self.action_space,
            'observation_space': self.observation_space,
            'global_state_dim': self.global_state_dim,
            'max_steps': self.config.max_steps,
            'episode_limit': self.config.max_steps,
            'obs_shape': self.observation_space.shape[0],
            'n_actions': self.action_space.n,
            'state_shape': self.global_state_dim
        }

    def _log_episode_stats(self) -> None:
        """Log episode statistics"""
        stats = {
            'episode': self.total_episodes,
            'step': self.game_state.current_step,
            'termination_reason': self.game_state.termination_reason,
            'total_reward': self.game_state.total_reward,
            'vip_hp': self.game_state.vip.hp,
            'threats_killed': self.game_state.stats['threats_killed'],
            'agents_killed': self.game_state.stats['agents_killed'],
            'messages_sent': self.game_state.stats['messages_sent'],
            'body_blocks': self.game_state.stats['body_blocks'],
            'long_range_kills': self.game_state.stats['long_range_kills'],
        }

        self.episode_stats.append(stats)

        logger.info(f"Episode {self.total_episodes} completed: "
                   f"Reason: {stats['termination_reason']}, "
                   f"Steps: {stats['step']}, "
                   f"Reward: {stats['total_reward']:.2f}, "
                   f"VIP HP: {stats['vip_hp']}, "
                   f"Threats killed: {stats['threats_killed']}")

    def close(self) -> None:
        """Close the environment"""
        if self.renderer:
            self.renderer.close()


def create_dem_env(difficulty: str = "normal", **kwargs) -> DEMEnv:
    """
    Create a DEM environment with specified difficulty

    Args:
        difficulty: Difficulty level ("easy", "normal", "hard")
        **kwargs: Additional configuration parameters

    Returns:
        DEM environment instance
    """
    config = DEMConfig(difficulty=difficulty, **kwargs)
    return DEMEnv(config)


# Predefined environment configurations
def create_dem_env_easy(**kwargs) -> DEMEnv:
    """Create easy difficulty DEM environment"""
    return create_dem_env(difficulty="easy", **kwargs)


def create_dem_env_normal(**kwargs) -> DEMEnv:
    """Create normal difficulty DEM environment"""
    return create_dem_env(difficulty="normal", **kwargs)


def create_dem_env_hard(**kwargs) -> DEMEnv:
    """Create hard difficulty DEM environment"""
    return create_dem_env(difficulty="hard", **kwargs)