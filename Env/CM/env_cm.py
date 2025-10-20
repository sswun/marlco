"""
CM (Collaborative Moving) Environment

A multi-agent collaborative box pushing environment where agents must work together
to move a box to a target location. This environment is designed to be simple yet
effective for testing multi-agent reinforcement learning algorithms.
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Any
import gymnasium as gym
from gymnasium import spaces

from .core import Agent, Box, Goal, Position, CMGameState, ActionType
from .config import CMConfig, get_config_by_name


class CooperativeMovingEnv(gym.Env):
    """
    Multi-agent collaborative box pushing environment.

    Agents need to cooperate to push a box from its initial position to a target
    location. The box can only be moved successfully when multiple agents push
    from different sides, with success probability increasing with cooperation.
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}

    def __init__(self, config: Optional[CMConfig] = None, difficulty: str = "normal", **kwargs):
        """
        Initialize the CM environment.

        Args:
            config: Environment configuration object
            difficulty: Predefined difficulty level ("easy", "normal", "hard")
            **kwargs: Additional configuration overrides
        """
        super().__init__()

        # Load configuration
        if config is None:
            config = get_config_by_name(difficulty)

        # Apply any configuration overrides
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        self.config = config
        self.n_agents = config.n_agents
        self.agent_ids = config.agent_ids

        # Set random seed if provided
        if config.seed is not None:
            np.random.seed(config.seed)
            random.seed(config.seed)

        # Action and observation spaces
        self.n_actions = 5  # STAY, UP, DOWN, LEFT, RIGHT
        self.action_space = spaces.Discrete(self.n_actions)

        # Calculate observation dimension
        # [self_x, self_y, box_center_x, box_center_y, goal_center_x, goal_center_y,
        #  relative_positions_of_other_agents (2 * (n_agents - 1))]
        obs_dim = 6 + 2 * (self.n_agents - 1)
        self.observation_space = spaces.Box(
            low=0.0, high=float(config.grid_size),
            shape=(obs_dim,), dtype=np.float32
        )

        # Environment state
        self.game_state: Optional[CMGameState] = None
        self.current_step = 0

        # Enhanced reward tracking
        self._last_box_distance = None
        self._last_agent_positions = {}
        self._steps_without_progress = 0
        self._first_agent_at_box = False

        # Rendering
        self.render_mode = config.render_mode
        self.fig = None
        self.ax = None

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Dict[str, np.ndarray]:
        """
        Reset the environment to initial state.

        Returns:
            observations: Dictionary mapping agent IDs to observations
        """
        super().reset(seed=seed)

        # Set seed if provided
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.current_step = 0

        # Initialize enhanced reward tracking
        self._last_box_distance = None
        self._last_agent_positions = {}
        self._steps_without_progress = 0
        self._first_agent_at_box = False

        # Initialize box position (center area)
        box_x = self.config.grid_size // 2 - 1
        box_y = self.config.grid_size // 2 - 1
        box_pos = Position(box_x, box_y)

        # Initialize goal position (random but not overlapping with box)
        goal_pos = self._generate_goal_position(box_pos)
        goal = Goal(goal_pos, self.config.goal_size)

        # Initialize agents
        agents = self._initialize_agents(box_pos)

        # Create game state
        box = Box(box_pos, self.config.box_size)
        self.game_state = CMGameState(agents, box, goal, self.config.grid_size)

        # Get initial observations
        observations = self._get_observations()

        return observations

    def step(self, actions: Dict[str, int]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            actions: Dictionary mapping agent IDs to action indices

        Returns:
            observations: New observations for all agents
            rewards: Rewards for all agents
            dones: Done flags for each agent
            info: Additional environment information
        """
        if self.game_state is None:
            raise RuntimeError("Environment must be reset before calling step")

        self.current_step += 1

        # Validate actions
        if self.agent_ids and set(actions.keys()) != set(self.agent_ids):
            raise ValueError(f"Actions must be provided for all agents. Expected: {self.agent_ids}")

        # Convert actions to ActionType
        action_dict = {}
        for agent_id, action_idx in actions.items():
            if not (0 <= action_idx < self.n_actions):
                raise ValueError(f"Invalid action {action_idx} for agent {agent_id}")
            action_dict[agent_id] = ActionType(action_idx)

        # 1. Move agents
        collision_penalty = self._move_agents(action_dict)

        # 2. Handle box pushing
        box_moved = self._handle_box_pushing(action_dict)

        # 3. Calculate rewards
        rewards = self._calculate_rewards(collision_penalty, box_moved)

        # 4. Check episode termination
        terminated = self.game_state.is_complete()
        truncated = self.current_step >= self.config.max_steps

        # 5. Get observations and info
        observations = self._get_observations()
        info = self._get_info()

        # Add step-specific info
        info.update({
            'step': self.current_step,
            'box_moved': box_moved,
            'collision_penalty': collision_penalty,
            'pushing_agents': self.game_state.get_pushing_agents(),
            'push_sides': self.game_state.get_push_sides(),
            'terminated': terminated,
            'truncated': truncated
        })

        # Combine terminated and truncated for consistency with other environments
        done = terminated or truncated
        
        # Create dones dictionary - all agents have same episode status (for consistency with DEM/HRG/MSFS)
        dones = {agent_id: done for agent_id in self.agent_ids}

        return observations, rewards, dones, info

    def _generate_goal_position(self, box_pos: Position) -> Position:
        """Generate a goal position that doesn't overlap with the box."""
        while True:
            goal_x = np.random.randint(0, self.config.grid_size - self.config.goal_size)
            goal_y = np.random.randint(0, self.config.grid_size - self.config.goal_size)
            goal_pos = Position(goal_x, goal_y)

            # Ensure goal is not too close to initial box position
            distance = abs(goal_x - box_pos.x) + abs(goal_y - box_pos.y)
            if distance >= 3:
                return goal_pos

    def _initialize_agents(self, box_pos: Position) -> List[Agent]:
        """Initialize agents around the box."""
        agents = []
        occupied_positions = set()

        # Add box positions to occupied set
        for dx in range(self.config.box_size):
            for dy in range(self.config.box_size):
                occupied_positions.add((box_pos.x + dx, box_pos.y + dy))

        # Generate valid positions around the box
        valid_positions = []
        for dx in range(-1, self.config.box_size + 1):
            for dy in range(-1, self.config.box_size + 1):
                x, y = box_pos.x + dx, box_pos.y + dy
                if (0 <= x < self.config.grid_size and
                    0 <= y < self.config.grid_size and
                    (x, y) not in occupied_positions):
                    valid_positions.append(Position(x, y))

        # Shuffle positions for randomness
        np.random.shuffle(valid_positions)

        # Place agents
        for i in range(self.n_agents):
            if i < len(valid_positions):
                pos = valid_positions[i]
            else:
                # If not enough positions around box, place randomly
                pos = self._find_random_empty_position(occupied_positions)

            agent = Agent(self.agent_ids[i], pos)
            agents.append(agent)
            occupied_positions.add((pos.x, pos.y))

        return agents

    def _find_random_empty_position(self, occupied_positions: set) -> Position:
        """Find a random empty position in the grid."""
        while True:
            x = np.random.randint(0, self.config.grid_size)
            y = np.random.randint(0, self.config.grid_size)
            if (x, y) not in occupied_positions:
                return Position(x, y)

    def _move_agents(self, actions: Dict[str, ActionType]) -> float:
        """Move all agents and handle collisions."""
        collision_penalty = 0.0
        new_positions = {}

        # First, calculate new positions
        for agent_id, action in actions.items():
            agent = self.game_state.get_agent(agent_id)
            old_pos = agent.position.copy()
            success = agent.move(action, self.config.grid_size)
            new_positions[agent_id] = agent.position

            # Check if agent hit boundary
            if not success:
                collision_penalty += self.config.box_collision_penalty

        # Check for agent-agent collisions
        position_agents = {}
        for agent_id, pos in new_positions.items():
            pos_key = (pos.x, pos.y)
            if pos_key in position_agents:
                # Collision detected
                collision_penalty += self.config.agent_collision_penalty

                # Reset both agents to old positions
                self.game_state.get_agent(agent_id).position = self.game_state.get_agent(agent_id).position
                self.game_state.get_agent(position_agents[pos_key]).position = self.game_state.get_agent(position_agents[pos_key]).position
            else:
                position_agents[pos_key] = agent_id

        # Check for agent-box collisions
        for agent_id, agent in self.game_state.agents.items():
            if self._is_agent_in_box(agent.position, self.game_state.box):
                # Agent can't move into box, revert to old position
                # This is handled by not updating the position in the first place
                pass

        return collision_penalty

    def _is_agent_in_box(self, pos: Position, box: Box) -> bool:
        """Check if agent position is inside the box."""
        return (box.position.x <= pos.x < box.position.x + box.size and
                box.position.y <= pos.y < box.position.y + box.size)

    def _handle_box_pushing(self, actions: Dict[str, ActionType]) -> bool:
        """Handle box pushing mechanics."""
        box = self.game_state.box

        # Get agents that are pushing the box
        pushing_sides = self.game_state.get_push_sides()
        n_cooperating = len(pushing_sides)

        if n_cooperating == 0:
            return False

        # Get success probability based on number of cooperating agents
        success_prob = self.config.push_success_probs.get(n_cooperating, 0.0)

        # Check if push is successful
        if np.random.random() < success_prob:
            # Calculate push direction
            push_direction = self.game_state.calculate_push_direction(pushing_sides)

            # Check if box can move in that direction
            if box.can_move(push_direction, self.config.grid_size):
                box.move(push_direction)
                return True

        return False

    def _calculate_rewards(self, collision_penalty: float, box_moved: bool) -> Dict[str, float]:
        """Calculate selective rewards to discourage random exploration."""
        # Base time penalty (stronger now)
        reward = self.config.time_penalty + collision_penalty

        # Get current positions and distances
        box_center = np.array(self.game_state.box.get_center())
        goal_center = np.array(self.game_state.goal.get_center())
        current_box_distance = np.linalg.norm(box_center - goal_center)

        # 1. Distance improvement reward - only for meaningful progress
        if self._last_box_distance is not None:
            distance_improvement = self._last_box_distance - current_box_distance
            # Higher threshold for meaningful improvement
            if distance_improvement > 0.2:  # Only reward significant progress
                reward += distance_improvement * self.config.distance_reward_scale

        # Update last distance for next step
        self._last_box_distance = current_box_distance

        # 2. Box movement reward - only if movement is toward goal
        if box_moved and hasattr(self, '_prev_box_position'):
            prev_center = np.array([
                self._prev_box_position.x + self.config.box_size / 2,
                self._prev_box_position.y + self.config.box_size / 2
            ])
            prev_distance = np.linalg.norm(prev_center - goal_center)

            # Only reward movement toward goal
            if current_box_distance < prev_distance:
                reward += self.config.box_move_reward_scale

        # 3. Cooperation reward - only when actually pushing together
        pushing_sides = self.game_state.get_push_sides()
        n_pushing = len(pushing_sides)

        if n_pushing > 1:
            cooperation_bonus = self.config.cooperation_reward * (n_pushing - 1)
            reward += cooperation_bonus

        # 4. Goal reached reward - main reward signal
        if self.game_state.is_complete():
            reward += self.config.goal_reached_reward

            # Efficiency bonus for quick completion
            if self.current_step < self.config.max_steps * 0.5:
                reward += 15.0  # Bonus for fast completion
            elif self.current_step < self.config.max_steps * 0.7:
                reward += 5.0  # Smaller bonus

        # Store current box position for next step
        self._prev_box_position = self.game_state.box.position.copy()

        # All agents receive the same reward (team reward)
        return {agent_id: reward for agent_id in self.agent_ids}

    def _get_observations(self) -> Dict[str, np.ndarray]:
        """Get observations for all agents."""
        observations = {}

        box_center = np.array(self.game_state.box.get_center())
        goal_center = np.array(self.game_state.goal.get_center())

        for agent_id, agent in self.game_state.agents.items():
            obs = []

            # Agent's own position
            obs.extend([float(agent.position.x), float(agent.position.y)])

            # Box center position
            obs.extend(box_center)

            # Goal center position
            obs.extend(goal_center)

            # Relative positions of other agents
            for other_id, other_agent in self.game_state.agents.items():
                if other_id != agent_id:
                    rel_x = other_agent.position.x - agent.position.x
                    rel_y = other_agent.position.y - agent.position.y
                    obs.extend([float(rel_x), float(rel_y)])

            # Convert to numpy array
            obs_array = np.array(obs, dtype=np.float32)

            # Normalize if configured
            if self.config.normalize_observations:
                obs_array = obs_array / float(self.config.grid_size)

            observations[agent_id] = obs_array

        return observations

    def _get_info(self) -> Dict[str, Any]:
        """Get additional environment information."""
        box_center = self.game_state.box.get_center()
        goal_center = self.game_state.goal.get_center()
        distance = np.linalg.norm(np.array(box_center) - np.array(goal_center))

        return {
            'box_position': (self.game_state.box.position.x, self.game_state.box.position.y),
            'goal_position': (self.game_state.goal.position.x, self.game_state.goal.position.y),
            'box_center': box_center,
            'goal_center': goal_center,
            'distance_to_goal': distance,
            'agents_complete': self.game_state.is_complete(),
            'pushing_agents': self.game_state.get_pushing_agents(),
            'push_sides': self.game_state.get_push_sides()
        }

    def get_avail_actions(self, agent_id: str) -> List[int]:
        """Get available actions for a specific agent."""
        if self.game_state is None:
            raise RuntimeError("Environment must be reset before getting available actions")

        agent = self.game_state.get_agent(agent_id)
        if agent is None:
            raise ValueError(f"Agent {agent_id} not found")

        return agent.get_valid_actions(self.config.grid_size)

    def render(self) -> Optional[np.ndarray]:
        """Render the environment."""
        if self.render_mode is None:
            return None

        if self.game_state is None:
            return None

        # Import matplotlib here to avoid issues with headless systems
        try:
            import matplotlib
            # Use non-interactive backend for headless systems
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            matplotlib_available = True
        except ImportError:
            print("Warning: Matplotlib not available, rendering disabled")
            return None
        except Exception as e:
            print(f"Warning: Matplotlib setup failed: {e}")
            return None

        if not matplotlib_available:
            return None

        if self.fig is None:
            try:
                self.fig, self.ax = plt.subplots(figsize=(8, 8))
                if self.render_mode == "human":
                    # For human mode, try to use interactive backend
                    try:
                        plt.switch_backend('TkAgg')
                        plt.ion()
                    except:
                        # Fall back to Agg backend
                        pass
            except Exception as e:
                print(f"Warning: Could not create matplotlib figure: {e}")
                return None

        self.ax.clear()
        self.ax.set_xlim(-0.5, self.config.grid_size - 0.5)
        self.ax.set_ylim(-0.5, self.config.grid_size - 0.5)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xticks(range(self.config.grid_size))
        self.ax.set_yticks(range(self.config.grid_size))

        # Draw goal area
        goal_rect = patches.Rectangle(
            (self.game_state.goal.position.y - 0.4, self.game_state.goal.position.x - 0.4),
            self.config.goal_size + 0.8, self.config.goal_size + 0.8,
            linewidth=2, edgecolor='green', facecolor='lightgreen', alpha=0.3
        )
        self.ax.add_patch(goal_rect)

        # Draw box
        box_rect = patches.Rectangle(
            (self.game_state.box.position.y - 0.3, self.game_state.box.position.x - 0.3),
            self.config.box_size + 0.6, self.config.box_size + 0.6,
            linewidth=2, edgecolor='brown', facecolor='orange', alpha=0.7
        )
        self.ax.add_patch(box_rect)

        # Draw agents
        colors = ['red', 'blue', 'purple', 'cyan', 'magenta', 'yellow']
        for i, (agent_id, agent) in enumerate(self.game_state.agents.items()):
            color = colors[i % len(colors)]
            circle = patches.Circle(
                (agent.position.y, agent.position.x), 0.3,
                color=color, alpha=0.8
            )
            self.ax.add_patch(circle)
            self.ax.text(agent.position.y, agent.position.x, str(i + 1),
                        ha='center', va='center', color='white', fontweight='bold')

        # Set title
        distance = self._get_info()['distance_to_goal']
        self.ax.set_title(f'Step: {self.current_step} | Distance: {distance:.2f}', fontsize=14)

        # Invert y-axis to match grid coordinates
        self.ax.invert_yaxis()

        if self.render_mode == 'human':
            plt.draw()
            try:
                plt.pause(0.1)
                plt.show(block=False)
            except:
                # Fallback if interactive display fails
                pass
        elif self.render_mode == 'rgb_array':
            try:
                self.fig.canvas.draw()
                data = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
                data = data.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
                return data
            except Exception as e:
                print(f"Warning: Could not generate rgb array: {e}")
                return None

        return None

    def close(self):
        """Close the environment and clean up resources."""
        if self.fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self.fig)
            self.fig = None
            self.ax = None

    def get_env_info(self) -> Dict[str, Any]:
        """Get environment information useful for training algorithms."""
        return {
            'n_agents': self.n_agents,
            'agent_ids': self.agent_ids,
            'n_actions': self.n_actions,
            'obs_dims': {agent_id: self.observation_space.shape[0] for agent_id in self.agent_ids},
            'act_dims': {agent_id: self.n_actions for agent_id in self.agent_ids},
            'episode_limit': self.config.max_steps,
            'grid_size': self.config.grid_size,
            'box_size': self.config.box_size
        }


# Factory functions for easy environment creation
def create_cm_env(difficulty: str = "normal", **kwargs) -> CooperativeMovingEnv:
    """Create a CM environment with specified difficulty."""
    return CooperativeMovingEnv(difficulty=difficulty, **kwargs)


def create_cm_env_from_config(config: CMConfig) -> CooperativeMovingEnv:
    """Create a CM environment from a configuration object."""
    return CooperativeMovingEnv(config=config)


if __name__ == "__main__":
    # Simple test
    env = create_cm_env(difficulty="easy", render_mode="")

    print("Environment Info:")
    print(env.get_env_info())

    obs = env.reset()
    print(f"Initial observations: {list(obs.keys())}")

    # Run a few random steps
    for step in range(10):
        actions = {}
        for agent_id in env.agent_ids:
            avail_actions = env.get_avail_actions(agent_id)
            actions[agent_id] = np.random.choice(avail_actions)

        obs, rewards, dones, info = env.step(actions)
        
        print(f"Step {step + 1}: rewards={list(rewards.values())[0]:.3f}, distance={info['distance_to_goal']:.2f}, done={list(dones.values())[0]}")
        
        if list(dones.values())[0]:
            print(f"Episode ended at step {step + 1}, goal achieved: {info['agents_complete']}")
            break
    
    env.close()
    print("Test completed successfully!")