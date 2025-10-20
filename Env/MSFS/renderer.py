"""
MSFS Environment Renderer

This module provides visualization functionality for the MSFS environment,
allowing users to observe agent behaviors and manufacturing pipeline evolution.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import pygame
from typing import Dict, List, Tuple, Optional, Any
import logging

from .core import Agent, Order, OrderType, WorkstationType, ActionType, GameState
from .config import MSFSConfig

# Configure logging
logger = logging.getLogger(__name__)


class MSFSRenderer:
    """Renderer for MSFS environment using pygame for real-time visualization"""

    def __init__(self, config: MSFSConfig):
        """
        Initialize MSFS renderer

        Args:
            config: Environment configuration
        """
        self.config = config
        self.grid_size = config.render_grid_size
        self.fps = config.render_fps

        # Layout: 3 workstations side by side
        self.workstation_width = self.grid_size // 3
        self.workstation_height = self.grid_size // 2

        # Window dimensions
        self.window_width = self.grid_size
        self.window_height = self.grid_size + 150  # Extra space for stats

        # Colors (RGB)
        self.colors = {
            'background': (240, 240, 240),
            'workstation_bg': (220, 220, 220),
            'raw_station': (100, 150, 255),      # Blue
            'assembly_station': (255, 150, 100),   # Orange
            'packing_station': (100, 255, 150),   # Green
            'queue_bg': (200, 200, 200),
            'simple_order': (255, 255, 100),       # Yellow
            'complex_order': (255, 100, 255),      # Magenta
            'agent': (50, 50, 200),               # Dark blue
            'agent_moving': (150, 150, 255),       # Light blue
            'agent_specialized': (255, 100, 100),  # Red for specialized agents
            'text': (0, 0, 0),
            'grid': (180, 180, 180),
            'stats_bg': (250, 250, 250),
            'progress_bar': (0, 200, 0),
            'progress_bg': (200, 200, 200)
        }

        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("MSFS Environment - Smart Manufacturing Flow Scheduling")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        self.tiny_font = pygame.font.Font(None, 14)

        # Animation
        self.animation_timer = 0

    def render(self, game_state: GameState, mode: str = "human"):
        """
        Render the current game state

        Args:
            game_state: Current game state to render
            mode: Rendering mode ("human" or "rgb_array")
        """
        self.screen.fill(self.colors['background'])

        # Draw workstations
        self._draw_workstations(game_state)

        # Draw orders in queues
        self._draw_orders(game_state)

        # Draw agents
        self._draw_agents(game_state)

        # Draw statistics panel
        self._draw_statistics_panel(game_state)

        # Handle display based on mode
        if mode == "human":
            pygame.display.flip()
            self.clock.tick(self.fps)
        elif mode == "rgb_array":
            # Return numpy array of the screen
            return self._get_rgb_array()

        self.animation_timer += 1

    def _draw_workstations(self, game_state: GameState):
        """Draw workstation areas"""
        for i, (ws_type, workstation) in enumerate(game_state.workstations.items()):
            x = i * self.workstation_width
            y = 50

            # Workstation background
            color = {
                WorkstationType.RAW: self.colors['raw_station'],
                WorkstationType.ASSEMBLY: self.colors['assembly_station'],
                WorkstationType.PACKING: self.colors['packing_station']
            }.get(ws_type, self.colors['workstation_bg'])

            # Draw workstation area
            rect = pygame.Rect(x, y, self.workstation_width - 10, self.workstation_height)
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, self.colors['grid'], rect, 2)

            # Draw workstation label
            label = {
                WorkstationType.RAW: "RAW",
                WorkstationType.ASSEMBLY: "ASSEMBLY",
                WorkstationType.PACKING: "PACKING"
            }.get(ws_type, "STATION")

            text = self.font.render(label, True, self.colors['text'])
            text_rect = text.get_rect(center=(x + self.workstation_width // 2 - 5, y + 20))
            self.screen.blit(text, text_rect)

            # Draw queue area
            queue_y = y + self.workstation_height + 10
            queue_height = 80
            queue_rect = pygame.Rect(x, queue_y, self.workstation_width - 10, queue_height)
            pygame.draw.rect(self.screen, self.colors['queue_bg'], queue_rect)
            pygame.draw.rect(self.screen, self.colors['grid'], queue_rect, 1)

            # Queue label
            queue_label = f"Queue: {len(workstation.queue)}"
            if workstation.current_order:
                queue_label += f" (+1 processing)"
            queue_text = self.small_font.render(queue_label, True, self.colors['text'])
            queue_text_rect = queue_text.get_rect(center=(x + self.workstation_width // 2 - 5, queue_y + 10))
            self.screen.blit(queue_text, queue_text_rect)

    def _draw_orders(self, game_state: GameState):
        """Draw orders in queues"""
        for i, (ws_type, workstation) in enumerate(game_state.workstations.items()):
            x = i * self.workstation_width
            queue_y = 50 + self.workstation_height + 10

            # Draw orders in queue
            max_visible = min(8, len(workstation.queue))
            for j in range(max_visible):
                order = workstation.queue[j]
                order_x = x + 10 + (j % 4) * 30
                order_y = queue_y + 30 + (j // 4) * 25

                # Order color based on type
                color = self.colors['simple_order'] if order.order_type == OrderType.SIMPLE else self.colors['complex_order']

                # Draw order as small rectangle
                order_rect = pygame.Rect(order_x, order_y, 25, 20)
                pygame.draw.rect(self.screen, color, order_rect)
                pygame.draw.rect(self.screen, self.colors['grid'], order_rect, 1)

                # Draw order type label
                order_type = "S" if order.order_type == OrderType.SIMPLE else "C"
                order_text = self.tiny_font.render(order_type, True, self.colors['text'])
                order_text_rect = order_text.get_rect(center=(order_x + 12, order_y + 10))
                self.screen.blit(order_text, order_text_rect)

            # Draw current processing order
            if workstation.current_order:
                process_x = x + self.workstation_width // 2 - 5 - 15
                process_y = 50 + self.workstation_height // 2 - 10

                color = self.colors['simple_order'] if workstation.current_order.order_type == OrderType.SIMPLE else self.colors['complex_order']
                process_rect = pygame.Rect(process_x, process_y, 30, 25)
                pygame.draw.rect(self.screen, color, process_rect)
                pygame.draw.rect(self.screen, self.colors['grid'], process_rect, 2)

                # Progress bar
                processing_time = workstation.current_order.get_processing_time(ws_type)
                progress = workstation.current_order.processing_progress / max(1, processing_time)
                progress_rect = pygame.Rect(process_x, process_y + 28, 30, 4)
                pygame.draw.rect(self.screen, self.colors['progress_bg'], progress_rect)
                if progress > 0:
                    progress_fill = pygame.Rect(process_x, process_y + 28, int(30 * progress), 4)
                    pygame.draw.rect(self.screen, self.colors['progress_bar'], progress_fill)

    def _draw_agents(self, game_state: GameState):
        """Draw agents"""
        for agent_id, agent in game_state.agents.items():
            # Determine agent position based on workstation
            ws_index = list(game_state.workstations.keys()).index(agent.current_workstation)
            agent_x = ws_index * self.workstation_width + self.workstation_width // 2 - 5

            # Agent position in workstation
            agent_y = 50 + self.workstation_height // 2 + 20

            # Agent color based on state
            if agent.move_cooldown > 0:
                color = self.colors['agent_moving']
            elif max(agent.consecutive_specialization.values()) >= 3:
                color = self.colors['agent_specialized']
            else:
                color = self.colors['agent']

            # Draw agent
            agent_rect = pygame.Rect(agent_x - 15, agent_y, 30, 30)
            pygame.draw.circle(self.screen, color, (agent_x, agent_y + 15), 15)
            pygame.draw.circle(self.screen, self.colors['grid'], (agent_x, agent_y + 15), 15, 2)

            # Draw agent ID
            if self.config.show_agent_ids:
                agent_text = self.tiny_font.render(agent_id.split('_')[-1], True, (255, 255, 255))
                agent_text_rect = agent_text.get_rect(center=(agent_x, agent_y + 15))
                self.screen.blit(agent_text, agent_text_rect)

            # Draw carried order
            if agent.carrying_order:
                carry_x = agent_x + 20
                carry_y = agent_y

                carry_color = self.colors['simple_order'] if agent.carrying_order.order_type == OrderType.SIMPLE else self.colors['complex_order']
                carry_rect = pygame.Rect(carry_x, carry_y, 20, 15)
                pygame.draw.rect(self.screen, carry_color, carry_rect)
                pygame.draw.rect(self.screen, self.colors['grid'], carry_rect, 1)

                carry_type = "S" if agent.carrying_order.order_type == OrderType.SIMPLE else "C"
                carry_text = self.tiny_font.render(carry_type, True, self.colors['text'])
                carry_text_rect = carry_text.get_rect(center=(carry_x + 10, carry_y + 7))
                self.screen.blit(carry_text, carry_text_rect)

            # Draw specialization indicator
            max_consecutive = max(agent.consecutive_specialization.values())
            if max_consecutive > 0:
                spec_text = self.tiny_font.render(f"×{max_consecutive}", True, color)
                spec_text_rect = spec_text.get_rect(center=(agent_x, agent_y - 10))
                self.screen.blit(spec_text, spec_text_rect)

            # Draw move cooldown
            if agent.move_cooldown > 0:
                cd_text = self.tiny_font.render(f"CD:{agent.move_cooldown}", True, self.colors['text'])
                cd_text_rect = cd_text.get_rect(center=(agent_x, agent_y + 45))
                self.screen.blit(cd_text, cd_text_rect)

    def _draw_statistics_panel(self, game_state: GameState):
        """Draw statistics panel at the bottom"""
        panel_y = self.window_height - 140

        # Draw panel background
        pygame.draw.rect(self.screen, self.colors['stats_bg'],
                        (0, panel_y, self.window_width, 140))
        pygame.draw.line(self.screen, self.colors['grid'],
                        (0, panel_y), (self.window_width, panel_y), 2)

        # Draw statistics
        stats = [
            f"Step: {game_state.current_step}/{game_state.max_steps}",
            f"Orders: {game_state.orders_completed}/{game_state.total_orders_generated}",
            f"S/C: {game_state.simple_orders_completed}/{game_state.complex_orders_completed}",
            f"Reward: {game_state.total_reward:.1f}",
            f"Specialization Events: {game_state.specialization_events}"
        ]

        x_offset = 10
        y_offset = panel_y + 10

        for i, stat in enumerate(stats):
            text = self.small_font.render(stat, True, self.colors['text'])
            self.screen.blit(text, (x_offset, y_offset))
            x_offset += 150
            if x_offset > self.window_width - 200:
                x_offset = 10
                y_offset += 25

        # Draw phase information
        phase_text = ""
        if game_state.current_step < 15:
            phase_text = "Phase 1: Warm-up (Low arrival, 30% complex)"
        elif game_state.current_step < 35:
            phase_text = "Phase 2: Peak (High arrival, 60% complex)"
        else:
            phase_text = "Phase 3: Wind-down (Low arrival, 40% complex)"

        phase_surface = self.small_font.render(phase_text, True, (100, 100, 100))
        self.screen.blit(phase_surface, (10, panel_y + 60))

        # Draw utilization info
        utilization_text = "Station Utilization: "
        for ws_type in WorkstationType:
            utilization = game_state.station_utilization[ws_type] / max(1, game_state.current_step)
            utilization_text += f"{ws_type.name[0]}:{utilization:.1%} "

        util_surface = self.small_font.render(utilization_text, True, (100, 100, 100))
        self.screen.blit(util_surface, (10, panel_y + 85))

        # Draw role emergence info
        if game_state.agents:
            role_text = "Agent Specialization: "
            for agent_id, agent in game_state.agents.items():
                max_spec = max(agent.specialization_count.values())
                if max_spec > 0:
                    best_ws = max(agent.specialization_count, key=agent.specialization_count.get)
                    role_text += f"{agent_id.split('_')[-1]}:{best_ws.name[0]}({max_spec}) "
                else:
                    role_text += f"{agent_id.split('_')[-1]}:None "

            role_surface = self.tiny_font.render(role_text, True, (100, 100, 100))
            self.screen.blit(role_surface, (10, panel_y + 110))

    def _get_rgb_array(self) -> np.ndarray:
        """Get current screen as RGB array"""
        try:
            # Convert pygame surface to numpy array
            view = pygame.surfarray.array3d(self.screen)
            # Transpose to get RGB format
            return np.transpose(view, (1, 0, 2))
        except:
            # Fallback to simple array
            return np.zeros((self.window_height, self.window_width, 3), dtype=np.uint8)

    def close(self):
        """Close the renderer"""
        pygame.quit()