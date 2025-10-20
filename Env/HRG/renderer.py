"""
HRG Environment Renderer

This module provides visualization functionality for the HRG environment,
allowing users to observe agent behaviors and game state evolution.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import pygame
from typing import Dict, List, Tuple, Optional, Any
import logging

from .core import AgentType, ResourceType, GameState, Position

# Configure logging
logger = logging.getLogger(__name__)


class HRGRenderer:
    """Renderer for HRG environment using pygame for real-time visualization"""

    def __init__(self, grid_size: int = 10, cell_size: int = 60, fps: int = 4):
        """
        Initialize HRG renderer

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
            'obstacle': (80, 80, 80),
            'base': (100, 150, 200),
            'scout': (100, 200, 100),
            'worker': (200, 150, 100),
            'transporter': (200, 100, 200),
            'gold': (255, 215, 0),
            'wood': (139, 69, 19),
            'text': (0, 0, 0),
            'vision_range': (200, 200, 255, 50)
        }

        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("HRG Environment Visualization")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 16)

        # Render options
        self.show_vision_ranges = True
        self.show_agent_ids = True
        self.show_inventory = True

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

        # Draw obstacles
        self._draw_obstacles(game_state)

        # Draw base
        self._draw_base()

        # Draw resources
        self._draw_resources(game_state)

        # Draw agents
        self._draw_agents(game_state)

        # Draw vision ranges (optional)
        if self.show_vision_ranges:
            self._draw_vision_ranges(game_state)

        # Draw statistics
        self._draw_statistics(game_state)

        # Update display
        if mode == "human":
            pygame.display.flip()
            self.clock.tick(self.fps)

        return self._get_rgb_array() if mode == "rgb_array" else None

    def _draw_grid(self):
        """Draw grid lines"""
        for i in range(self.grid_size + 1):
            # Vertical lines
            pygame.draw.line(
                self.screen, self.colors['grid'],
                (i * self.cell_size, 0),
                (i * self.cell_size, self.window_size)
            )
            # Horizontal lines
            pygame.draw.line(
                self.screen, self.colors['grid'],
                (0, i * self.cell_size),
                (self.window_size, i * self.cell_size)
            )

    def _draw_obstacles(self, game_state: GameState):
        """Draw obstacles on the grid"""
        for obstacle_pos in game_state.obstacles:
            rect = pygame.Rect(
                obstacle_pos.x * self.cell_size,
                obstacle_pos.y * self.cell_size,
                self.cell_size,
                self.cell_size
            )
            pygame.draw.rect(self.screen, self.colors['obstacle'], rect)
            pygame.draw.rect(self.screen, (0, 0, 0), rect, 2)  # Border

    def _draw_base(self):
        """Draw the base at position (0, 0)"""
        base_rect = pygame.Rect(0, 0, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, self.colors['base'], base_rect)
        pygame.draw.rect(self.screen, (0, 0, 100), base_rect, 3)  # Thick border

        # Draw "B" for base
        text = self.font.render("B", True, self.colors['text'])
        text_rect = text.get_rect(center=(self.cell_size // 2, self.cell_size // 2))
        self.screen.blit(text, text_rect)

    def _draw_resources(self, game_state: GameState):
        """Draw resources on the grid"""
        for resource in game_state.resources:
            if not resource.is_active:
                continue

            x = resource.position.x * self.cell_size
            y = resource.position.y * self.cell_size

            # Choose color based on resource type
            if resource.resource_type == ResourceType.GOLD:
                color = self.colors['gold']
                symbol = "G"
            else:
                color = self.colors['wood']
                symbol = "W"

            # Draw resource circle
            center = (x + self.cell_size // 2, y + self.cell_size // 2)
            radius = self.cell_size // 4
            pygame.draw.circle(self.screen, color, center, radius)
            pygame.draw.circle(self.screen, (0, 0, 0), center, radius, 2)

            # Draw quantity
            text = self.small_font.render(str(resource.remaining_quantity), True, self.colors['text'])
            text_rect = text.get_rect(center=center)
            self.screen.blit(text, text_rect)

    def _draw_agents(self, game_state: GameState):
        """Draw agents on the grid"""
        for agent in game_state.agents.values():
            x = agent.position.x * self.cell_size
            y = agent.position.y * self.cell_size

            # Choose color based on agent type
            if agent.type == AgentType.SCOUT:
                color = self.colors['scout']
                symbol = "S"
            elif agent.type == AgentType.WORKER:
                color = self.colors['worker']
                symbol = "W"
            else:  # TRANSPORTER
                color = self.colors['transporter']
                symbol = "T"

            # Draw agent rectangle
            agent_rect = pygame.Rect(
                x + 5, y + 5,
                self.cell_size - 10, self.cell_size - 10
            )
            pygame.draw.rect(self.screen, color, agent_rect)
            pygame.draw.rect(self.screen, (0, 0, 0), agent_rect, 2)

            # Draw agent symbol
            text = self.font.render(symbol, True, self.colors['text'])
            text_rect = text.get_rect(center=(
                x + self.cell_size // 2,
                y + self.cell_size // 2 - 5
            ))
            self.screen.blit(text, text_rect)

            # Draw inventory if enabled
            if self.show_inventory and agent.is_carrying_resources:
                inventory_text = f"G:{agent.inventory[ResourceType.GOLD]} W:{agent.inventory[ResourceType.WOOD]}"
                text = self.small_font.render(inventory_text, True, self.colors['text'])
                text_rect = text.get_rect(center=(
                    x + self.cell_size // 2,
                    y + self.cell_size - 10
                ))
                self.screen.blit(text, text_rect)

            # Draw energy bar
            energy_ratio = agent.energy / 100.0
            bar_width = self.cell_size - 10
            bar_height = 4
            bar_x = x + 5
            bar_y = y + 1

            # Background
            pygame.draw.rect(self.screen, (255, 0, 0),
                           (bar_x, bar_y, bar_width, bar_height))
            # Energy level
            pygame.draw.rect(self.screen, (0, 255, 0),
                           (bar_x, bar_y, int(bar_width * energy_ratio), bar_height))

    def _draw_vision_ranges(self, game_state: GameState):
        """Draw vision ranges for agents"""
        # Create a transparent surface
        vision_surface = pygame.Surface((self.window_size, self.window_size), pygame.SRCALPHA)

        for agent in game_state.agents.values():
            # Calculate vision range in pixels
            vision_range_cells = agent.config.vision_range
            vision_range_pixels = vision_range_cells * self.cell_size

            # Draw vision circle
            center_x = agent.position.x * self.cell_size + self.cell_size // 2
            center_y = agent.position.y * self.cell_size + self.cell_size // 2

            pygame.draw.circle(
                vision_surface,
                self.colors['vision_range'],
                (center_x, center_y),
                vision_range_pixels
            )

        self.screen.blit(vision_surface, (0, 0))

    def _draw_statistics(self, game_state: GameState):
        """Draw game statistics"""
        stats = [
            f"Step: {game_state.current_step}/{game_state.max_steps}",
            f"Score: {game_state.total_score:.1f}",
            f"Gold: {game_state.deposited_resources[ResourceType.GOLD]}",
            f"Wood: {game_state.deposited_resources[ResourceType.WOOD]}",
        ]

        y_offset = 10
        for stat in stats:
            text = self.small_font.render(stat, True, self.colors['text'])
            self.screen.blit(text, (10, y_offset))
            y_offset += 20

    def _get_rgb_array(self) -> np.ndarray:
        """Get current screen as RGB array"""
        # Convert pygame surface to RGB array
        width, height = self.screen.get_size()
        rgb_array = pygame.surfarray.array3d(self.screen)
        # Transpose from (width, height, channels) to (height, width, channels)
        rgb_array = rgb_array.transpose((1, 0, 2))
        return rgb_array

    def toggle_vision_ranges(self):
        """Toggle vision range display"""
        self.show_vision_ranges = not self.show_vision_ranges

    def toggle_agent_ids(self):
        """Toggle agent ID display"""
        self.show_agent_ids = not self.show_agent_ids

    def toggle_inventory(self):
        """Toggle inventory display"""
        self.show_inventory = not self.show_inventory

    def save_screenshot(self, filename: str):
        """Save current screen as image"""
        pygame.image.save(self.screen, filename)
        logger.info(f"Screenshot saved to {filename}")

    def close(self):
        """Close the renderer"""
        pygame.quit()


class MatplotlibRenderer:
    """Alternative renderer using matplotlib for static visualization"""

    def __init__(self, grid_size: int = 10, figsize: Tuple[int, int] = (8, 8)):
        """
        Initialize matplotlib renderer

        Args:
            grid_size: Size of the grid
            figsize: Figure size in inches
        """
        self.grid_size = grid_size
        self.figsize = figsize
        self.fig = None
        self.ax = None

        # Colors
        self.colors = {
            'obstacle': '#505050',
            'base': '#6496C8',
            'scout': '#64C864',
            'worker': '#C89664',
            'transporter': '#C864C8',
            'gold': '#FFD700',
            'wood': '#8B4513',
            'empty': 'white'
        }

    def render(self, game_state: GameState, save_path: Optional[str] = None):
        """
        Render the game state using matplotlib

        Args:
            game_state: Current game state
            save_path: Path to save the figure (optional)
        """
        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=self.figsize)

        # Create grid
        grid = np.full((self.grid_size, self.grid_size), 'empty', dtype=object)

        # Place obstacles
        for obstacle in game_state.obstacles:
            grid[obstacle.y, obstacle.x] = 'obstacle'

        # Place base
        grid[0, 0] = 'base'

        # Place resources
        for resource in game_state.resources:
            if resource.is_active:
                if resource.resource_type == ResourceType.GOLD:
                    grid[resource.position.y, resource.position.x] = 'gold'
                else:
                    grid[resource.position.y, resource.position.x] = 'wood'

        # Create color map
        color_map = np.vectorize(lambda x: self.colors[x])(grid)

        # Plot grid
        im = self.ax.imshow(color_map, extent=[0, self.grid_size, 0, self.grid_size])

        # Plot agents
        for agent in game_state.agents.values():
            x = agent.position.x + 0.5
            y = self.grid_size - agent.position.y - 0.5  # Flip y-axis

            if agent.type == AgentType.SCOUT:
                marker = 'o'
                color = self.colors['scout']
                label = 'S'
            elif agent.type == AgentType.WORKER:
                marker = 's'
                color = self.colors['worker']
                label = 'W'
            else:  # TRANSPORTER
                marker = '^'
                color = self.colors['transporter']
                label = 'T'

            self.ax.scatter(x, y, s=200, c=color, marker=marker,
                          edgecolors='black', linewidths=2, zorder=5)
            self.ax.text(x, y, label, ha='center', va='center',
                        fontsize=12, fontweight='bold', zorder=6)

        # Set grid properties
        self.ax.set_xlim(0, self.grid_size)
        self.ax.set_ylim(0, self.grid_size)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xticks(range(self.grid_size + 1))
        self.ax.set_yticks(range(self.grid_size + 1))
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_title(f'HRG Environment - Step {game_state.current_step}')

        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.colors['scout'],
                      markersize=10, label='Scout'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=self.colors['worker'],
                      markersize=10, label='Worker'),
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor=self.colors['transporter'],
                      markersize=10, label='Transporter'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.colors['gold'],
                      markersize=10, label='Gold'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.colors['wood'],
                      markersize=10, label='Wood'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=self.colors['base'],
                      markersize=10, label='Base'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=self.colors['obstacle'],
                      markersize=10, label='Obstacle')
        ]
        self.ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))

        # Add statistics
        stats_text = (
            f"Score: {game_state.total_score:.1f}\n"
            f"Gold: {game_state.deposited_resources[ResourceType.GOLD]}\n"
            f"Wood: {game_state.deposited_resources[ResourceType.WOOD]}\n"
            f"Step: {game_state.current_step}/{game_state.max_steps}"
        )
        self.ax.text(1.02, 0.5, stats_text, transform=self.ax.transAxes,
                    fontsize=10, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def close(self):
        """Close the matplotlib renderer"""
        if self.fig:
            plt.close(self.fig)


# Utility function for quick rendering
def render_game_state(game_state: GameState,
                     renderer_type: str = "pygame",
                     save_path: Optional[str] = None,
                     **kwargs):
    """
    Convenience function to render a game state

    Args:
        game_state: Game state to render
        renderer_type: Type of renderer ("pygame" or "matplotlib")
        save_path: Path to save the rendering (for matplotlib)
        **kwargs: Additional renderer parameters
    """
    if renderer_type == "pygame":
        renderer = HRGRenderer(**kwargs)
        renderer.render(game_state)
        # Keep window open for a moment
        pygame.time.wait(1000)
        renderer.close()
    else:
        renderer = MatplotlibRenderer(**kwargs)
        renderer.render(game_state, save_path)
        renderer.close()