"""
MSFS Environment Implementation

This module implements the Smart Manufacturing Flow Scheduling environment
with role emergence mechanisms for multi-agent reinforcement learning.
"""

import numpy as np
import random
import time
import logging
from typing import Dict, List, Tuple, Optional, Any

import gymnasium as gym
from gymnasium import spaces

from .core import (
    Order, OrderType, Workstation, WorkstationType, Agent, ActionType, GameState
)
from .config import MSFSConfig, MSFSPresetConfigs, get_config_by_name

# Configure logging
logger = logging.getLogger(__name__)


class MSFSEnv(gym.Env):
    """
    Smart Manufacturing Flow Scheduling Environment

    A multi-agent environment where agents (robots) must collaborate to process
    orders through a manufacturing pipeline, naturally forming roles through
    specialized reward signals.
    """

    def __init__(self, config: MSFSConfig = None):
        """
        Initialize MSFS environment

        Args:
            config: Environment configuration
        """
        self.config = config if config is not None else MSFSConfig()
        self.config.validate()

        # Set random seed
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
            random.seed(self.config.seed)

        # Initialize game state
        self.game_state = GameState(max_steps=self.config.max_steps)
        self._initialize_agents()
        self._initialize_workstations()

        # Initialize observation and action spaces
        self._init_spaces()

        # Episode tracking
        self.episode_start_time = None
        self.total_episodes = 0

        # Statistics tracking
        self.episode_stats = []

        # Initialize renderer
        self.renderer = None
        self._setup_renderer()

        logger.info(f"MSFS Environment initialized with {self.config.num_agents} agents")

    def _initialize_agents(self) -> None:
        """Initialize agents in the environment"""
        self.game_state.agents.clear()

        # Distribute agents across workstations initially
        workstation_distribution = [
            WorkstationType.RAW,
            WorkstationType.ASSEMBLY,
            WorkstationType.PACKING
        ]

        for i in range(self.config.num_agents):
            agent_id = f"agent_{i}"
            # Distribute agents evenly across workstations
            workstation = workstation_distribution[i % len(workstation_distribution)]

            agent = Agent(
                agent_id=agent_id,
                current_workstation=workstation
            )

            # Initialize tracking attributes for reward system
            agent._last_action = None
            agent._last_action_invalid = False
            agent._last_handoff = False
            agent._prepared_station = False
            agent._role_switch = False

            self.game_state.agents[agent_id] = agent

    def _initialize_workstations(self) -> None:
        """Initialize workstations"""
        self.game_state.workstations = {
            WorkstationType.RAW: Workstation(WorkstationType.RAW),
            WorkstationType.ASSEMBLY: Workstation(WorkstationType.ASSEMBLY),
            WorkstationType.PACKING: Workstation(WorkstationType.PACKING)
        }

    def _init_spaces(self) -> None:
        """Initialize observation and action spaces"""
        # Each agent has 8 discrete actions
        self.action_space = spaces.Discrete(8)

        # Calculate observation space dimension (24 dimensions per agent)
        # Self state: 10 dimensions (3 for workstation one-hot, 1 for move cooldown,
        #                  1 for carrying status, 5 for order stage/type info)
        # Global info: 7 dimensions (3 for queue lengths, 2 for order counts, 2 for time)
        # Teammate info: 7 dimensions (3 for workstation one-hot, 1 for busy status,
        #                   3 for carrying/processing info)
        obs_dim = 24

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Global state space for centralized training
        self.global_state_dim = 42  # Expanded global state representation

    def reset(self) -> Dict[str, np.ndarray]:
        """Reset environment for new episode"""
        # Reset game state
        self.game_state = GameState(max_steps=self.config.max_steps)
        self._initialize_agents()
        self._initialize_workstations()

        # Reset episode tracking
        self.episode_start_time = time.time()
        self.total_episodes += 1

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
        if self.game_state.is_terminated():
            observations = self.get_observations()
            rewards = self._get_zero_rewards()
            dones = {agent_id: True for agent_id in self.game_state.agents.keys()}
            return observations, rewards, dones, {}

        # Store previous state for reward calculation
        prev_stats = self._get_previous_stats()

        # Generate new orders
        self._generate_orders()

        # Update agent cooldowns
        self._update_agent_cooldowns()

        # Execute agent actions
        self._execute_agent_actions(actions)

        # Update workstation processing
        self._update_workstation_processing()

        # Update station utilization
        if self.config.track_utilization:
            self._update_utilization()

        # Check termination conditions
        terminated = self.game_state.is_terminated()

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

    def _get_previous_stats(self) -> Dict[str, float]:
        """Get previous statistics for reward calculation"""
        return {
            'orders_completed': self.game_state.orders_completed,
            'simple_orders_completed': self.game_state.simple_orders_completed,
            'complex_orders_completed': self.game_state.complex_orders_completed,
            'total_reward': self.game_state.total_reward,
            'efficiency_penalty': self.game_state.efficiency_penalty
        }

    def _generate_orders(self) -> None:
        """Generate new orders based on arrival probability"""
        new_order = self.game_state.generate_order()
        if new_order:
            # Add to raw material station queue
            raw_station = self.game_state.workstations[WorkstationType.RAW]
            if raw_station.add_order(new_order):
                self.game_state.orders.append(new_order)
                logger.debug(f"Generated new order: {new_order.order_id} ({new_order.order_type.name})")

    def _update_agent_cooldowns(self) -> None:
        """Update agent move cooldowns"""
        for agent in self.game_state.agents.values():
            agent.update_cooldown()

    def _execute_agent_actions(self, actions: Dict[str, int]) -> None:
        """Execute actions for all agents"""
        for agent_id, action in actions.items():
            if agent_id not in self.game_state.agents:
                continue

            agent = self.game_state.agents[agent_id]

            try:
                action_type = ActionType(action)
            except ValueError:
                logger.warning(f"Invalid action {action} for agent {agent_id}")
                continue

            self._execute_single_agent_action(agent, action_type)

    def _execute_single_agent_action(self, agent: Agent, action: ActionType) -> None:
        """Execute action for a single agent"""
        # Track action for reward calculation
        agent._last_action = action
        agent._last_action_invalid = False

        if action == ActionType.WAIT:
            return
        elif action in [ActionType.MOVE_TO_RAW, ActionType.MOVE_TO_ASSEMBLY, ActionType.MOVE_TO_PACKING]:
            self._execute_agent_move(agent, action)
        elif action == ActionType.PULL_ORDER:
            self._execute_agent_pull_order(agent)
        elif action == ActionType.START_PROCESSING:
            self._execute_agent_start_processing(agent)
        elif action == ActionType.COMPLETE_STAGE:
            self._execute_agent_complete_stage(agent)
        elif action == ActionType.DELIVER_ORDER:
            self._execute_agent_deliver_order(agent)
        else:
            # Invalid action
            agent._last_action_invalid = True

    def _execute_agent_move(self, agent: Agent, action: ActionType) -> None:
        """Execute movement action for agent"""
        if not agent.can_move():
            return

        target_workstation = {
            ActionType.MOVE_TO_RAW: WorkstationType.RAW,
            ActionType.MOVE_TO_ASSEMBLY: WorkstationType.ASSEMBLY,
            ActionType.MOVE_TO_PACKING: WorkstationType.PACKING
        }.get(action)

        if target_workstation and agent.current_workstation != target_workstation:
            # Track role switching (moving to different workstation type)
            if agent.current_workstation != target_workstation:
                agent._role_switch = True

            agent.move_to(target_workstation)
            logger.debug(f"{agent.agent_id} moved to {target_workstation.name}")

    def _execute_agent_pull_order(self, agent: Agent) -> None:
        """Execute order pulling action"""
        if agent.current_workstation != WorkstationType.RAW:
            return

        if agent.carrying_order is not None:
            return

        raw_station = self.game_state.workstations[WorkstationType.RAW]
        next_order = raw_station.get_next_order()

        if next_order:
            agent.carrying_order = next_order
            next_order.current_stage = 1  # In Raw station
            logger.debug(f"{agent.agent_id} pulled order {next_order.order_id}")

    def _execute_agent_start_processing(self, agent: Agent) -> None:
        """Execute start processing action"""
        if agent.carrying_order is None:
            return

        if agent.carrying_order.position != agent.current_workstation:
            return

        # Start or continue processing
        agent.carrying_order.processing_progress += 1

        processing_time = agent.carrying_order.get_processing_time(agent.current_workstation)

        if agent.carrying_order.processing_progress >= processing_time:
            # Processing complete
            logger.debug(f"{agent.agent_id} completed processing at {agent.current_workstation.name}")

    def _execute_agent_complete_stage(self, agent: Agent) -> None:
        """Execute stage completion action"""
        if agent.carrying_order is None:
            return

        if agent.carrying_order.processing_progress < agent.carrying_order.get_processing_time(agent.current_workstation):
            return

        # Move to next workstation
        current_stage = agent.carrying_order.current_stage
        next_stage = current_stage + 1

        if next_stage <= 3:  # Can move to next workstation
            next_workstation = {
                1: WorkstationType.ASSEMBLY,
                2: WorkstationType.PACKING,
                3: WorkstationType.PACKING  # Stays at packing for delivery
            }.get(next_stage)

            if next_workstation:
                next_station = self.game_state.workstations[next_workstation]
                if next_station.add_order(agent.carrying_order):
                    agent.carrying_order.current_stage = next_stage
                    agent.carrying_order.processing_progress = 0

                    # Track cooperation: preparing station for other agents
                    if next_station.get_queue_length() == 1:  # This agent made the station ready
                        agent._prepared_station = True

                    # Track cooperation: successful handoff
                    if agent.current_workstation != next_workstation:
                        agent._last_handoff = True

                    # Track specialization
                    agent.specialization_count[agent.current_workstation] += 1
                    agent.consecutive_specialization[agent.current_workstation] += 1

                    # Reset other specializations
                    for ws in WorkstationType:
                        if ws != agent.current_workstation:
                            agent.consecutive_specialization[ws] = 0

                    # Check for specialization reward
                    if (agent.consecutive_specialization[agent.current_workstation] >= self.config.specialization_threshold and
                        agent.consecutive_specialization[agent.current_workstation] == self.config.specialization_threshold):
                        self.game_state.specialization_events += 1

                    if next_stage == 4:  # Delivered
                        self.game_state.orders_completed += 1
                        if agent.carrying_order.order_type == OrderType.SIMPLE:
                            self.game_state.simple_orders_completed += 1
                        else:
                            self.game_state.complex_orders_completed += 1
                        self.game_state.completed_orders.append(agent.carrying_order)

                    logger.debug(f"{agent.agent_id} moved order {agent.carrying_order.order_id} to stage {next_stage}")

                    agent.carrying_order = None if next_stage < 4 else agent.carrying_order

    def _execute_agent_deliver_order(self, agent: Agent) -> None:
        """Execute order delivery action"""
        if agent.current_workstation != WorkstationType.PACKING:
            return

        if agent.carrying_order is None:
            return

        if agent.carrying_order.current_stage == 3:  # At packing station
            agent.carrying_order.current_stage = 4  # Delivered
            self.game_state.orders_completed += 1

            if agent.carrying_order.order_type == OrderType.SIMPLE:
                self.game_state.simple_orders_completed += 1
            else:
                self.game_state.complex_orders_completed += 1

            self.game_state.completed_orders.append(agent.carrying_order)

            # Track specialization
            agent.specialization_count[WorkstationType.PACKING] += 1
            agent.consecutive_specialization[WorkstationType.PACKING] += 1

            # Reset other specializations
            for ws in WorkstationType:
                if ws != WorkstationType.PACKING:
                    agent.consecutive_specialization[ws] = 0

            logger.debug(f"{agent.agent_id} delivered order {agent.carrying_order.order_id}")
            agent.carrying_order = None

    def _update_workstation_processing(self) -> None:
        """Update workstation processing progress"""
        for workstation in self.game_state.workstations.values():
            if workstation.current_order:
                # Progress is handled by agent actions
                pass

    def _update_utilization(self) -> None:
        """Update workstation utilization statistics"""
        for workstation_type, workstation in self.game_state.workstations.items():
            if workstation.get_queue_length() > 0:
                self.game_state.station_utilization[workstation_type] += 1.0

    def _calculate_rewards(self, prev_stats: Dict[str, float], terminated: bool) -> Dict[str, float]:
        """
        Enhanced reward calculation system for better exploration

        This system provides dense, immediate feedback for meaningful actions
        to encourage random exploration and learning.
        """
        rewards = {agent_id: 0.0 for agent_id in self.game_state.agents.keys()}

        # Track agent actions for immediate feedback
        action_rewards = self._calculate_action_rewards()

        # Track progress rewards for milestones
        progress_rewards = self._calculate_progress_rewards(prev_stats)

        # Track cooperation rewards for teamwork
        cooperation_rewards = self._calculate_cooperation_rewards()

        # Track role emergence rewards
        role_rewards = self._calculate_role_rewards()

        # Apply light penalties only for invalid actions
        penalty_rewards = self._calculate_penalties()

        # Combine all reward types
        for agent_id in rewards.keys():
            rewards[agent_id] = (
                action_rewards.get(agent_id, 0.0) +
                progress_rewards.get(agent_id, 0.0) +
                cooperation_rewards.get(agent_id, 0.0) +
                role_rewards.get(agent_id, 0.0) +
                penalty_rewards.get(agent_id, 0.0)
            )

        self.game_state.total_reward += sum(rewards.values())
        return rewards

    def _calculate_action_rewards(self) -> Dict[str, float]:
        """Calculate immediate rewards for agent actions"""
        rewards = {}

        for agent_id, agent in self.game_state.agents.items():
            reward = 0.0

            # Track actions from previous step using agent state
            if hasattr(agent, '_last_action') and agent._last_action is not None:
                action = agent._last_action

                # Movement rewards (if moving toward productive target)
                if action in [ActionType.MOVE_TO_RAW, ActionType.MOVE_TO_ASSEMBLY, ActionType.MOVE_TO_PACKING]:
                    target_station = {
                        ActionType.MOVE_TO_RAW: WorkstationType.RAW,
                        ActionType.MOVE_TO_ASSEMBLY: WorkstationType.ASSEMBLY,
                        ActionType.MOVE_TO_PACKING: WorkstationType.PACKING
                    }.get(action)

                    # Reward moving to station with work
                    if target_station:
                        station = self.game_state.workstations[target_station]
                        if station.get_queue_length() > 0 or agent.carrying_order:
                            reward += self.config.move_toward_target

                # Pickup rewards
                elif action == ActionType.PULL_ORDER and agent.carrying_order is not None:
                    reward += self.config.pickup_material

                # Processing rewards
                elif action == ActionType.START_PROCESSING and agent.carrying_order is not None:
                    processing_time = agent.carrying_order.get_processing_time(agent.current_workstation)
                    if agent.carrying_order.processing_progress == 1:  # Just started
                        reward += self.config.start_processing
                    elif agent.carrying_order.processing_progress >= processing_time:  # Just finished
                        reward += self.config.complete_stage

                # Stage completion rewards
                elif action == ActionType.COMPLETE_STAGE and agent.carrying_order is not None:
                    if agent.carrying_order.current_stage <= 3:
                        reward += self.config.complete_stage

                # Delivery rewards
                elif action == ActionType.DELIVER_ORDER:
                    reward += self.config.deliver_order

            rewards[agent_id] = reward

        return rewards

    def _calculate_progress_rewards(self, prev_stats: Dict[str, float]) -> Dict[str, float]:
        """Calculate milestone-based progress rewards"""
        rewards = {agent_id: 0.0 for agent_id in self.game_state.agents.keys()}

        # Track order progress
        orders_completed = self.game_state.orders_completed - prev_stats['orders_completed']
        simple_completed = self.game_state.simple_orders_completed - prev_stats['simple_orders_completed']
        complex_completed = self.game_state.complex_orders_completed - prev_stats['complex_orders_completed']

        team_progress_reward = 0.0

        # Order completion progress rewards
        if orders_completed > 0:
            # Base completion rewards
            team_progress_reward += simple_completed * self.config.simple_order_value
            team_progress_reward += complex_completed * self.config.complex_order_value

            # Enhanced progress rewards
            team_progress_reward += simple_completed * self.config.raw_completion    # RAW stage
            team_progress_reward += simple_completed * self.config.assembly_completion  # ASSEMBLY stage
            team_progress_reward += simple_completed * self.config.packaging_completion # PACKING stage
            team_progress_reward += simple_completed * self.config.order_delivery     # Final delivery

            # Extra rewards for complex orders
            team_progress_reward += complex_completed * (self.config.raw_completion * 1.5)
            team_progress_reward += complex_completed * (self.config.assembly_completion * 1.5)
            team_progress_reward += complex_completed * (self.config.packaging_completion * 1.5)
            team_progress_reward += complex_completed * (self.config.order_delivery * 1.5)

        # Smooth workflow bonuses
        # Check for efficient transitions between stations
        active_stations = sum(1 for ws in self.game_state.workstations.values()
                             if ws.get_queue_length() > 0 or ws.current_order)
        if active_stations >= 2:
            team_progress_reward += self.config.concurrent_processing

        # No queue buildup bonus
        total_queue_length = sum(ws.get_queue_length() for ws in self.game_state.workstations.values())
        if total_queue_length <= len(self.game_state.agents):
            team_progress_reward += self.config.no_queue_bonus

        # Distribute progress rewards equally
        if self.game_state.agents:
            reward_per_agent = team_progress_reward / len(self.game_state.agents)
            for agent_id in rewards:
                rewards[agent_id] = reward_per_agent

        return rewards

    def _calculate_cooperation_rewards(self) -> Dict[str, float]:
        """Calculate cooperation-based rewards"""
        rewards = {agent_id: 0.0 for agent_id in self.game_state.agents.keys()}

        team_cooperation_reward = 0.0

        # Track successful handoffs between agents
        for agent in self.game_state.agents.values():
            if hasattr(agent, '_last_handoff') and agent._last_handoff:
                team_cooperation_reward += self.config.successful_handoff
                agent._last_handoff = False  # Reset flag

        # Track workstation preparation for other agents
        for agent in self.game_state.agents.values():
            if hasattr(agent, '_prepared_station') and agent._prepared_station:
                team_cooperation_reward += self.config.workstation_ready
                agent._prepared_station = False  # Reset flag

        # Check for balanced workload distribution
        if len(self.game_state.agents) > 1:
            agents_per_station = len(self.game_state.agents) // 3
            station_distribution = {}
            for agent in self.game_state.agents.values():
                station = agent.current_workstation
                station_distribution[station] = station_distribution.get(station, 0) + 1

            # Reward balanced distribution
            if len(set(station_distribution.values())) <= 2:  # Relatively balanced
                team_cooperation_reward += self.config.balanced_workload

        # Distribute cooperation rewards equally
        if self.game_state.agents:
            reward_per_agent = team_cooperation_reward / len(self.game_state.agents)
            for agent_id in rewards:
                rewards[agent_id] = reward_per_agent

        return rewards

    def _calculate_role_rewards(self) -> Dict[str, float]:
        """Calculate role emergence rewards"""
        rewards = {agent_id: 0.0 for agent_id in self.game_state.agents.keys()}

        if not self.config.enable_role_emergence_rewards:
            return rewards

        for agent_id, agent in self.game_state.agents.items():
            role_reward = 0.0

            # Specialization focus rewards
            if agent.consecutive_specialization[WorkstationType.RAW] >= 2:
                role_reward += self.config.collector_focus
            if agent.consecutive_specialization[WorkstationType.ASSEMBLY] >= 2:
                role_reward += self.config.processor_focus
            if agent.consecutive_specialization[WorkstationType.PACKING] >= 2:
                role_reward += self.config.packager_focus

            # Role consistency rewards
            total_specializations = sum(agent.consecutive_specialization.values())
            max_specialization = max(agent.consecutive_specialization.values()) if agent.consecutive_specialization else 0
            if max_specialization >= 3 and max_specialization == total_specializations:
                role_reward += self.config.stick_to_role

            # Adaptive role switching rewards (when needed)
            if hasattr(agent, '_role_switch') and agent._role_switch:
                role_reward += self.config.switch_when_needed
                agent._role_switch = False  # Reset flag

            rewards[agent_id] = role_reward

        return rewards

    def _calculate_penalties(self) -> Dict[str, float]:
        """Calculate light penalties for exploration"""
        rewards = {agent_id: 0.0 for agent_id in self.game_state.agents.keys()}

        for agent_id, agent in self.game_state.agents.items():
            penalty = 0.0

            # Only penalize clearly invalid actions (very light penalty)
            if hasattr(agent, '_last_action_invalid') and agent._last_action_invalid:
                penalty += self.config.invalid_action_penalty
                agent._last_action_invalid = False  # Reset flag

            rewards[agent_id] = penalty

        return rewards

    def _get_zero_rewards(self) -> Dict[str, float]:
        """Get zero rewards for all agents"""
        return {agent_id: 0.0 for agent_id in self.game_state.agents.keys()}

    def _get_info(self, terminated: bool) -> Dict[str, Any]:
        """Get info dictionary"""
        info = {
            'episode_step': self.game_state.current_step,
            'max_steps': self.game_state.max_steps,
            'orders_completed': self.game_state.orders_completed,
            'simple_orders_completed': self.game_state.simple_orders_completed,
            'complex_orders_completed': self.game_state.complex_orders_completed,
            'total_orders_generated': self.game_state.total_orders_generated,
            'queue_stats': self.game_state.get_queue_stats(),
            'role_emergence_stats': self.game_state.get_role_emergence_stats()
        }

        if terminated:
            info['termination_reason'] = "max_steps_reached"
            info['total_reward'] = self.game_state.total_reward
            info['order_completion_rate'] = (
                self.game_state.orders_completed / max(1, self.game_state.total_orders_generated)
            )

        return info

    def get_observations(self) -> Dict[str, np.ndarray]:
        """Get observations for all agents"""
        observations = {}

        for agent_id, agent in self.game_state.agents.items():
            obs = self._get_agent_observation(agent)
            observations[agent_id] = obs

        return observations

    def _get_agent_observation(self, agent: Agent) -> np.ndarray:
        """Get observation for a single agent"""
        obs = np.zeros(24, dtype=np.float32)
        idx = 0

        # Self state (10 dimensions)
        # Current workstation (one-hot)
        obs[idx + agent.current_workstation.value] = 1.0
        idx += 3

        # Move cooldown (normalized)
        obs[idx] = agent.move_cooldown / max(1, self.config.move_cooldown_time)
        idx += 1

        # Carrying status
        if agent.carrying_order:
            obs[idx] = 1.0  # Is carrying
            idx += 1

            # Order type
            obs[idx] = 1.0 if agent.carrying_order.order_type == OrderType.SIMPLE else -1.0
            idx += 1

            # Current stage (normalized)
            obs[idx] = agent.carrying_order.current_stage / 4.0
            idx += 1

            # Processing progress (normalized)
            max_time = 3  # Maximum processing time
            obs[idx] = agent.carrying_order.processing_progress / max_time
            idx += 1
        else:
            idx += 5  # Skip carrying info

        # Specialization info (normalized)
        max_specialization = 10  # Expected max specialization count
        for ws in WorkstationType:
            obs[idx] = agent.consecutive_specialization[ws] / max_specialization
            idx += 1

        # Global information (7 dimensions)
        # Queue lengths (normalized)
        max_queue = self.config.queue_limit
        for ws in WorkstationType:
            queue_len = self.game_state.workstations[ws].get_queue_length()
            obs[idx] = queue_len / max_queue
            idx += 1

        # Order counts (normalized)
        total_orders = max(1, self.game_state.total_orders_generated)
        obs[idx] = self.game_state.simple_orders_completed / total_orders
        idx += 1
        obs[idx] = self.game_state.complex_orders_completed / total_orders
        idx += 1

        # Time information (normalized)
        obs[idx] = self.game_state.current_step / self.game_state.max_steps
        idx += 1

        # Teammate information (7 dimensions) - simplified for 2 agents
        other_agents = [a for a in self.game_state.agents.values() if a.agent_id != agent.agent_id]

        if other_agents:
            teammate = other_agents[0]

            # Teammate workstation (one-hot)
            obs[idx + teammate.current_workstation.value] = 1.0
            idx += 3

            # Teammate busy status
            obs[idx] = 1.0 if teammate.is_busy() else -1.0
            idx += 1

            # Teammate carrying status
            obs[idx] = 1.0 if teammate.carrying_order else -1.0
            idx += 1
        else:
            idx += 5  # No teammate info

        return obs

    def get_global_state(self) -> np.ndarray:
        """Get global state representation for centralized training"""
        state = np.zeros(self.global_state_dim, dtype=np.float32)
        idx = 0

        # Agent states (2 agents * 8 dimensions each)
        for agent in self.game_state.agents.values():
            # Workstation (one-hot)
            state[idx + agent.current_workstation.value] = 1.0
            idx += 3

            # Move cooldown (normalized)
            state[idx] = agent.move_cooldown / max(1, self.config.move_cooldown_time)
            idx += 1

            # Carrying status and order info
            if agent.carrying_order:
                state[idx] = 1.0
                idx += 1
                state[idx] = 1.0 if agent.carrying_order.order_type == OrderType.SIMPLE else -1.0
                idx += 1
                state[idx] = agent.carrying_order.current_stage / 4.0
                idx += 1
            else:
                idx += 4
        idx = 16  # Reset after agent info

        # Workstation states (3 stations * 6 dimensions each)
        for ws_type in WorkstationType:
            workstation = self.game_state.workstations[ws_type]

            # Queue length (normalized)
            state[idx] = workstation.get_queue_length() / self.config.queue_limit
            idx += 1

            # Order type distribution
            if workstation.queue:
                simple_count = sum(1 for o in workstation.queue if o.order_type == OrderType.SIMPLE)
                complex_count = len(workstation.queue) - simple_count
                state[idx] = simple_count / max(1, len(workstation.queue))
                idx += 1
                state[idx] = complex_count / max(1, len(workstation.queue))
                idx += 1
            else:
                idx += 2

            # Current order info
            if workstation.current_order:
                state[idx] = 1.0
                idx += 1
                state[idx] = 1.0 if workstation.current_order.order_type == OrderType.SIMPLE else -1.0
                idx += 1
                state[idx] = workstation.current_order.processing_progress / 3.0
                idx += 1
            else:
                idx += 3
        idx = 34  # Reset after workstation info

        # Global statistics (8 dimensions)
        state[idx] = self.game_state.current_step / self.game_state.max_steps
        idx += 1
        state[idx] = self.game_state.orders_completed / max(1, self.game_state.total_orders_generated)
        idx += 1
        state[idx] = self.game_state.simple_orders_completed / max(1, self.game_state.orders_completed)
        idx += 1
        state[idx] = self.game_state.complex_orders_completed / max(1, self.game_state.orders_completed)
        idx += 1
        state[idx] = min(1.0, self.game_state.total_reward / 100.0)  # Normalized reward
        idx += 1
        state[idx] = self.game_state.specialization_events / 10.0  # Normalized specialization
        idx += 1
        state[idx] = float(self.game_state.current_step >= self.config.finishing_phase_start)
        idx += 1

        return state

    def _setup_renderer(self):
        """Setup renderer based on render mode"""
        if self.config.render_mode in ["human", "rgb_array"]:
            try:
                from .renderer import MSFSRenderer
                self.renderer = MSFSRenderer(
                    config=self.config
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
            'total_reward': self.game_state.total_reward,
            'orders_completed': self.game_state.orders_completed,
            'simple_orders_completed': self.game_state.simple_orders_completed,
            'complex_orders_completed': self.game_state.complex_orders_completed,
            'total_orders_generated': self.game_state.total_orders_generated,
            'specialization_events': self.game_state.specialization_events,
            'order_completion_rate': (
                self.game_state.orders_completed / max(1, self.game_state.total_orders_generated)
            )
        }

        self.episode_stats.append(stats)

        logger.info(f"Episode {self.total_episodes} completed: "
                   f"Steps: {stats['step']}, "
                   f"Reward: {stats['total_reward']:.2f}, "
                   f"Orders: {stats['orders_completed']}/{stats['total_orders_generated']}, "
                   f"Completion Rate: {stats['order_completion_rate']:.2%}")

    def close(self) -> None:
        """Close the environment"""
        if self.renderer:
            self.renderer.close()


def create_msfs_env(difficulty: str = "normal", **kwargs) -> MSFSEnv:
    """
    Create a MSFS environment with specified difficulty

    Args:
        difficulty: Difficulty level ("easy", "normal", "hard")
        **kwargs: Additional configuration parameters

    Returns:
        MSFS environment instance
    """
    if difficulty == "easy":
        config = MSFSPresetConfigs.easy()
    elif difficulty == "normal":
        config = MSFSPresetConfigs.normal()
    elif difficulty == "hard":
        config = MSFSPresetConfigs.hard()
    else:
        config = MSFSConfig()

    # Apply any parameter overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"Warning: Unknown configuration parameter: {key}")

    return MSFSEnv(config)


# Predefined environment configurations
def create_msfs_env_easy(**kwargs) -> MSFSEnv:
    """Create easy difficulty MSFS environment"""
    return create_msfs_env(difficulty="easy", **kwargs)


def create_msfs_env_normal(**kwargs) -> MSFSEnv:
    """Create normal difficulty MSFS environment"""
    return create_msfs_env(difficulty="normal", **kwargs)


def create_msfs_env_hard(**kwargs) -> MSFSEnv:
    """Create hard difficulty MSFS environment"""
    return create_msfs_env(difficulty="hard", **kwargs)