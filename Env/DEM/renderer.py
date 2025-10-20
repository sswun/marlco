"""
DEM Environment Renderer

This module provides visualization functionality for the DEM environment,
allowing users to observe agent behaviors and game state evolution in the VIP escort mission.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import pygame
from typing import Dict, List, Tuple, Optional, Any
import logging

from .core import Agent, VIP, Threat, ActionType, MessageType, ThreatType, TerrainType, GameState, Position

# Configure logging
logger = logging.getLogger(__name__)


class DEMRenderer:
    """Renderer for DEM environment using pygame for real-time visualization"""

    def __init__(self, grid_size: int = 12, cell_size: int = 50, fps: int = 4):
        """
        Initialize DEM renderer

        Args:
            grid_size: Size of the grid
            cell_size: Size of each cell in pixels
            fps: Frames per second for animation
        """
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.fps = fps
        self.window_size = grid_size * cell_size

        # Colors (RGB)
        self.colors = {
            'background': (240, 240, 240),
            'grid': (200, 200, 200),
            'river': (100, 150, 255),
            'forest': (34, 139, 34),
            'vip': (255, 215, 0),
            'vip_target': (255, 0, 0),
            'agent': (0, 100, 200),
            'agent_guard': (0, 200, 100),
            'rusher': (255, 0, 0),
            'shooter': (255, 100, 0),
            'text': (0, 0, 0),
            'vision_range': (200, 200, 255, 50),
            'attack_range': (255, 200, 200, 30),
            'hp_bar': (0, 255, 0),
            'hp_bar_bg': (255, 0, 0),
            'message': (100, 100, 255)
        }

        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_size, self.window_size + 100))
        pygame.display.set_caption("DEM Environment Visualization - VIP Escort Mission")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 16)
        self.tiny_font = pygame.font.Font(None, 12)

        # Render options
        self.show_vision_ranges = False
        self.show_agent_ids = True
        self.show_hp_bars = True
        self.show_messages = True

        # Animation
        self.animation_timer = 0
        self.attack_animations = []

        # Statistics
        self.stats_surface = None
        self.stats_dirty = True

    def render(self, game_state: GameState, mode: str = "human"):
        """
        Render the current game state

        Args:
            game_state: Current game state to render
            mode: Rendering mode ("human" or "rgb_array")
        """
        self.screen.fill(self.colors['background'])

        # Draw grid
        self._draw_grid()

        # Draw terrain
        self._draw_terrain(game_state)

        # Draw VIP target
        self._draw_vip_target(game_state)

        # Draw VIP
        self._draw_vip(game_state)

        # Draw threats
        self._draw_threats(game_state)

        # Draw agents
        self._draw_agents(game_state)

        # Draw vision ranges (optional)
        if self.show_vision_ranges:
            self._draw_vision_ranges(game_state)

        # Draw attack ranges (optional)
        self._draw_attack_ranges(game_state)

        # Draw messages
        if self.show_messages:
            self._draw_messages(game_state)

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

    def _draw_grid(self):
        """Draw grid lines"""
        for x in range(self.grid_size + 1):
            pygame.draw.line(self.screen, self.colors['grid'],
                           (x * self.cell_size, 0),
                           (x * self.cell_size, self.window_size))
        for y in range(self.grid_size + 1):
            pygame.draw.line(self.screen, self.colors['grid'],
                           (0, y * self.cell_size),
                           (self.window_size, y * self.cell_size))

    def _draw_terrain(self, game_state: GameState):
        """Draw terrain features (rivers, forests)"""
        # Draw rivers
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if game_state.terrain[x, y] == TerrainType.RIVER:
                    rect = pygame.Rect(x * self.cell_size, y * self.cell_size,
                                     self.cell_size, self.cell_size)
                    pygame.draw.rect(self.screen, self.colors['river'], rect)
                    # Add wave pattern
                    wave_offset = (self.animation_timer // 2) % 4
                    for i in range(0, self.cell_size, 4):
                        pygame.draw.line(self.screen, (80, 130, 235),
                                       (x * self.cell_size + i, y * self.cell_size + wave_offset),
                                       (x * self.cell_size + i + 2, y * self.cell_size + wave_offset + 2), 1)

                elif game_state.terrain[x, y] == TerrainType.FOREST:
                    rect = pygame.Rect(x * self.cell_size, y * self.cell_size,
                                     self.cell_size, self.cell_size)
                    pygame.draw.rect(self.screen, self.colors['forest'], rect)
                    # Add tree pattern
                    tree_x = x * self.cell_size + self.cell_size // 2
                    tree_y = y * self.cell_size + self.cell_size // 2
                    pygame.draw.circle(self.screen, (0, 100, 0), (tree_x, tree_y), self.cell_size // 4)

    def _draw_vip_target(self, game_state: GameState):
        """Draw VIP target position"""
        x, y = game_state.vip.target_pos.x, game_state.vip.target_pos.y
        center_x = x * self.cell_size + self.cell_size // 2
        center_y = y * self.cell_size + self.cell_size // 2

        # Draw target with pulsing effect
        pulse = abs(np.sin(self.animation_timer * 0.1))
        radius = int(self.cell_size * 0.4 * (1 + pulse * 0.2))

        pygame.draw.circle(self.screen, self.colors['vip_target'], (center_x, center_y), radius, 2)
        pygame.draw.circle(self.screen, self.colors['vip_target'], (center_x, center_y), radius + 5, 1)

        # Draw target symbol
        font_size = int(self.cell_size * 0.4)
        font = pygame.font.Font(None, font_size)
        text = font.render("TARGET", True, self.colors['vip_target'])
        text_rect = text.get_rect(center=(center_x, center_y))
        self.screen.blit(text, text_rect)

    def _draw_vip(self, game_state: GameState):
        """Draw VIP"""
        if not game_state.vip.is_alive:
            return

        x, y = game_state.vip.pos.x, game_state.vip.pos.y
        center_x = x * self.cell_size + self.cell_size // 2
        center_y = y * self.cell_size + self.cell_size // 2

        # Draw VIP as a star
        points = []
        for i in range(10):
            angle = np.pi * i / 5
            if i % 2 == 0:
                r = self.cell_size * 0.4
            else:
                r = self.cell_size * 0.2
            px = center_x + r * np.cos(angle - np.pi / 2)
            py = center_y + r * np.sin(angle - np.pi / 2)
            points.append((px, py))

        pygame.draw.polygon(self.screen, self.colors['vip'], points)
        pygame.draw.polygon(self.screen, (200, 150, 0), points, 2)

        # Draw VIP label
        if self.show_agent_ids:
            text = self.small_font.render("VIP", True, self.colors['text'])
            text_rect = text.get_rect(center=(center_x, center_y - self.cell_size // 2 - 10))
            self.screen.blit(text, text_rect)

        # Draw HP bar
        if self.show_hp_bars:
            self._draw_hp_bar(center_x, center_y + self.cell_size // 2 + 5,
                            game_state.vip.hp, game_state.vip.max_hp)

    def _draw_agents(self, game_state: GameState):
        """Draw agents"""
        for agent_id, agent in game_state.agents.items():
            if not agent.is_alive:
                continue

            x, y = agent.pos.x, agent.pos.y
            center_x = x * self.cell_size + self.cell_size // 2
            center_y = y * self.cell_size + self.cell_size // 2

            # Determine agent color based on current action
            color = self.colors['agent']
            if hasattr(agent, 'last_action'):
                if agent.last_action == ActionType.GUARD:
                    color = self.colors['agent_guard']

            # Draw agent
            pygame.draw.circle(self.screen, color, (center_x, center_y), self.cell_size // 3)
            pygame.draw.circle(self.screen, (0, 50, 100), (center_x, center_y), self.cell_size // 3, 2)

            # Draw agent ID
            if self.show_agent_ids:
                text = self.tiny_font.render(agent_id.split('_')[-1], True, (255, 255, 255))
                text_rect = text.get_rect(center=(center_x, center_y))
                self.screen.blit(text, text_rect)

            # Draw HP bar
            if self.show_hp_bars:
                self._draw_hp_bar(center_x, center_y + self.cell_size // 3 + 5,
                                agent.hp, agent.max_hp)

    def _draw_threats(self, game_state: GameState):
        """Draw threats"""
        for threat in game_state.threats.values():
            if not threat.is_alive:
                continue

            x, y = threat.pos.x, threat.pos.y
            center_x = x * self.cell_size + self.cell_size // 2
            center_y = y * self.cell_size + self.cell_size // 2

            # Determine threat color based on type
            if threat.type == ThreatType.RUSHER:
                color = self.colors['rusher']
                shape = 'triangle'
            else:  # SHOOTER
                color = self.colors['shooter']
                shape = 'diamond'

            # Draw threat
            if shape == 'triangle':
                # Draw triangle for rusher
                points = [
                    (center_x, center_y - self.cell_size // 3),
                    (center_x - self.cell_size // 3, center_y + self.cell_size // 3),
                    (center_x + self.cell_size // 3, center_y + self.cell_size // 3)
                ]
                pygame.draw.polygon(self.screen, color, points)
                pygame.draw.polygon(self.screen, (150, 0, 0), points, 2)
            else:
                # Draw diamond for shooter
                points = [
                    (center_x, center_y - self.cell_size // 3),
                    (center_x + self.cell_size // 3, center_y),
                    (center_x, center_y + self.cell_size // 3),
                    (center_x - self.cell_size // 3, center_y)
                ]
                pygame.draw.polygon(self.screen, color, points)
                pygame.draw.polygon(self.screen, (150, 50, 0), points, 2)

            # Draw threat type label
            text = self.tiny_font.render(threat.type.value[0], True, (255, 255, 255))
            text_rect = text.get_rect(center=(center_x, center_y))
            self.screen.blit(text, text_rect)

            # Draw HP bar
            if self.show_hp_bars:
                self._draw_hp_bar(center_x, center_y + self.cell_size // 3 + 5,
                                threat.hp, threat.max_hp)

    def _draw_vision_ranges(self, game_state: GameState):
        """Draw vision ranges"""
        # VIP vision range
        if game_state.vip.is_alive:
            x, y = game_state.vip.pos.x, game_state.vip.pos.y
            center_x = x * self.cell_size + self.cell_size // 2
            center_y = y * self.cell_size + self.cell_size // 2
            radius = game_state.vip.vision_range * self.cell_size

            # Create surface with alpha
            vision_surface = pygame.Surface((self.window_size, self.window_size), pygame.SRCALPHA)
            pygame.draw.circle(vision_surface, self.colors['vision_range'],
                             (center_x, center_y), radius)
            self.screen.blit(vision_surface, (0, 0))

        # Agent vision ranges
        for agent in game_state.agents.values():
            if not agent.is_alive:
                continue
            x, y = agent.pos.x, agent.pos.y
            center_x = x * self.cell_size + self.cell_size // 2
            center_y = y * self.cell_size + self.cell_size // 2
            radius = agent.vision_range * self.cell_size

            vision_surface = pygame.Surface((self.window_size, self.window_size), pygame.SRCALPHA)
            pygame.draw.circle(vision_surface, self.colors['vision_range'],
                             (center_x, center_y), radius)
            self.screen.blit(vision_surface, (0, 0))

    def _draw_attack_ranges(self, game_state: GameState):
        """Draw attack ranges"""
        # Agent attack ranges
        for agent in game_state.agents.values():
            if not agent.is_alive:
                continue
            x, y = agent.pos.x, agent.pos.y
            center_x = x * self.cell_size + self.cell_size // 2
            center_y = y * self.cell_size + self.cell_size // 2
            radius = agent.range * self.cell_size

            attack_surface = pygame.Surface((self.window_size, self.window_size), pygame.SRCALPHA)
            pygame.draw.circle(attack_surface, self.colors['attack_range'],
                             (center_x, center_y), radius)
            self.screen.blit(attack_surface, (0, 0))

    def _draw_messages(self, game_state: GameState):
        """Draw communication messages"""
        recent_messages = game_state.messages[-5:] if len(game_state.messages) > 5 else game_state.messages

        y_offset = self.window_size + 10
        for message in recent_messages:
            if isinstance(message, dict):
                sender = message.get('sender', 'Unknown')
                msg_type = message.get('type', 'Unknown')
                text = self.tiny_font.render(f"{sender}: {msg_type}",
                                            True, self.colors['message'])
            else:
                text = self.tiny_font.render(f"Message", True, self.colors['message'])
            self.screen.blit(text, (10, y_offset))
            y_offset += 15

    def _draw_hp_bar(self, x: int, y: int, current_hp: int, max_hp: int):
        """Draw HP bar"""
        bar_width = self.cell_size // 2
        bar_height = 4

        # Background
        pygame.draw.rect(self.screen, self.colors['hp_bar_bg'],
                        (x - bar_width // 2, y, bar_width, bar_height))

        # Current HP
        hp_percentage = max(0, current_hp / max_hp)
        hp_width = int(bar_width * hp_percentage)
        if hp_width > 0:
            pygame.draw.rect(self.screen, self.colors['hp_bar'],
                           (x - bar_width // 2, y, hp_width, bar_height))

    def _draw_statistics_panel(self, game_state: GameState):
        """Draw statistics panel at the bottom"""
        panel_y = self.window_size

        # Draw panel background
        pygame.draw.rect(self.screen, (220, 220, 220),
                        (0, panel_y, self.window_size, 100))
        pygame.draw.line(self.screen, self.colors['grid'],
                        (0, panel_y), (self.window_size, panel_y), 2)

        # Draw statistics
        stats = [
            f"Step: {game_state.current_step}/{game_state.max_steps}",
            f"VIP HP: {game_state.vip.hp}/{game_state.vip.max_hp}",
            f"VIP Distance to Target: {game_state.stats['vip_distance_to_target']:.1f}",
            f"Agents Alive: {len([a for a in game_state.agents.values() if a.is_alive])}/{len(game_state.agents)}",
            f"Threats Alive: {len([t for t in game_state.threats.values() if t.is_alive])}",
            f"Messages: {len(game_state.messages)}"
        ]

        x_offset = 10
        y_offset = panel_y + 5

        for stat in stats:
            text = self.small_font.render(stat, True, self.colors['text'])
            self.screen.blit(text, (x_offset, y_offset))
            x_offset += 150
            if x_offset > self.window_size - 200:
                x_offset = 10
                y_offset += 20

    def _get_rgb_array(self) -> np.ndarray:
        """Get current screen as RGB array"""
        try:
            # Convert pygame surface to numpy array
            view = pygame.surfarray.array3d(self.screen)
            # Transpose to get RGB format
            return np.transpose(view, (1, 0, 2))
        except:
            # Fallback to simple array
            return np.zeros((self.window_size, self.window_size, 3), dtype=np.uint8)

    def close(self):
        """Close the renderer"""
        pygame.quit()

    def toggle_vision_ranges(self):
        """Toggle vision range display"""
        self.show_vision_ranges = not self.show_vision_ranges

    def toggle_hp_bars(self):
        """Toggle HP bar display"""
        self.show_hp_bars = not self.show_hp_bars

    def toggle_agent_ids(self):
        """Toggle agent ID display"""
        self.show_agent_ids = not self.show_agent_ids

    def toggle_messages(self):
        """Toggle message display"""
        self.show_messages = not self.show_messages