"""
CM Environment Renderer

This module provides visualization capabilities for the Collaborative Moving environment,
including both static and animated rendering options.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from typing import List, Dict, Optional, Any, Tuple
import os

from .core import CMGameState, Position


class MatplotlibRenderer:
    """Matplotlib-based renderer for the CM environment."""

    def __init__(self, grid_size: int = 7, figsize: Tuple[int, int] = (8, 8)):
        """
        Initialize the renderer.

        Args:
            grid_size: Size of the grid
            figsize: Figure size for rendering
        """
        self.grid_size = grid_size
        self.figsize = figsize
        self.fig = None
        self.ax = None

        # Color scheme
        self.colors = {
            'background': '#f0f0f0',
            'grid': '#cccccc',
            'box': '#ff8c00',
            'box_edge': '#8b4513',
            'goal': '#90ee90',
            'goal_edge': '#228b22',
            'agents': ['#ff0000', '#0000ff', '#800080', '#00ffff', '#ff00ff', '#ffff00'],
            'text': 'white',
            'title': 'black'
        }

    def render_frame(self, game_state: CMGameState, title: Optional[str] = None) -> plt.Figure:
        """
        Render a single frame of the environment.

        Args:
            game_state: Current game state
            title: Optional title for the plot

        Returns:
            matplotlib Figure object
        """
        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self._setup_plot()

        # Draw grid
        self._draw_grid()

        # Draw goal
        self._draw_goal(game_state.goal)

        # Draw box
        self._draw_box(game_state.box)

        # Draw agents
        self._draw_agents(game_state.agents)

        # Set title
        if title is None:
            distance = np.linalg.norm(
                np.array(game_state.box.get_center()) -
                np.array(game_state.goal.get_center())
            )
            title = f'Step: {game_state.current_step} | Distance: {distance:.2f}'

        self.ax.set_title(title, fontsize=14, color=self.colors['title'])

        # Add legend
        self._add_legend(game_state)

        return self.fig

    def _setup_plot(self):
        """Setup the basic plot properties."""
        self.ax.set_xlim(-0.5, self.grid_size - 0.5)
        self.ax.set_ylim(-0.5, self.grid_size - 0.5)
        self.ax.set_aspect('equal')
        self.ax.set_facecolor(self.colors['background'])

        # Set ticks
        self.ax.set_xticks(range(self.grid_size))
        self.ax.set_yticks(range(self.grid_size))

        # Invert y-axis to match grid coordinates
        self.ax.invert_yaxis()

    def _draw_grid(self):
        """Draw the grid lines."""
        for i in range(self.grid_size + 1):
            self.ax.axhline(y=i - 0.5, color=self.colors['grid'], linewidth=0.5)
            self.ax.axvline(x=i - 0.5, color=self.colors['grid'], linewidth=0.5)

    def _draw_goal(self, goal):
        """Draw the goal area."""
        goal_rect = patches.Rectangle(
            (goal.position.y - 0.4, goal.position.x - 0.4),
            goal.size + 0.8, goal.size + 0.8,
            linewidth=2, edgecolor=self.colors['goal_edge'],
            facecolor=self.colors['goal'], alpha=0.3
        )
        self.ax.add_patch(goal_rect)

        # Add goal label
        goal_center = goal.get_center()
        self.ax.text(goal_center[1], goal_center[0], 'GOAL',
                    ha='center', va='center', fontsize=12,
                    color=self.colors['goal_edge'], fontweight='bold')

    def _draw_box(self, box):
        """Draw the box."""
        box_rect = patches.Rectangle(
            (box.position.y - 0.3, box.position.x - 0.3),
            box.size + 0.6, box.size + 0.6,
            linewidth=2, edgecolor=self.colors['box_edge'],
            facecolor=self.colors['box'], alpha=0.7
        )
        self.ax.add_patch(box_rect)

        # Add box label
        box_center = box.get_center()
        self.ax.text(box_center[1], box_center[0], 'BOX',
                    ha='center', va='center', fontsize=10,
                    color=self.colors['text'], fontweight='bold')

    def _draw_agents(self, agents: Dict[str, Any]):
        """Draw all agents."""
        for i, (agent_id, agent) in enumerate(agents.items()):
            color = self.colors['agents'][i % len(self.colors['agents'])]

            # Draw agent circle
            circle = patches.Circle(
                (agent.position.y, agent.position.x), 0.3,
                color=color, alpha=0.8, edgecolor='black', linewidth=1
            )
            self.ax.add_patch(circle)

            # Add agent number
            self.ax.text(agent.position.y, agent.position.x, str(i + 1),
                        ha='center', va='center', color=self.colors['text'],
                        fontweight='bold', fontsize=10)

            # Add agent ID label
            self.ax.text(agent.position.y, agent.position.x + 0.5, agent_id,
                        ha='center', va='bottom', fontsize=8,
                        color=self.colors['title'])

    def _add_legend(self, game_state: CMGameState):
        """Add legend to the plot."""
        # Create legend elements
        legend_elements = []

        # Add pushing agents info
        pushing_agents = game_state.get_pushing_agents()
        if pushing_agents:
            legend_elements.append(plt.Line2D([0], [0], marker='*', color='w',
                                            markerfacecolor='gold', markersize=10,
                                            label=f'Pushing: {len(pushing_agents)} agents'))

        # Add distance info
        distance = np.linalg.norm(
            np.array(game_state.box.get_center()) -
            np.array(game_state.goal.get_center())
        )
        legend_elements.append(plt.Line2D([0], [0], color='gray',
                                        label=f'Distance: {distance:.2f}'))

        # Add completion status
        if game_state.is_complete():
            legend_elements.append(plt.Line2D([0], [0], color='green',
                                            linewidth=3, label='COMPLETE!'))

        if legend_elements:
            self.ax.legend(handles=legend_elements, loc='upper right',
                          bbox_to_anchor=(1.0, 1.0))

    def save_frame(self, filename: str, game_state: CMGameState, title: Optional[str] = None):
        """
        Save a single frame to file.

        Args:
            filename: Output filename
            game_state: Current game state
            title: Optional title
        """
        fig = self.render_frame(game_state, title)
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def close(self):
        """Close the renderer."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


class AnimationRenderer:
    """Renderer for creating animations of CM environment episodes."""

    def __init__(self, grid_size: int = 7, figsize: Tuple[int, int] = (10, 10)):
        """
        Initialize the animation renderer.

        Args:
            grid_size: Size of the grid
            figsize: Figure size for animation
        """
        self.grid_size = grid_size
        self.figsize = figsize
        self.base_renderer = MatplotlibRenderer(grid_size, figsize)
        self.frames = []

    def add_frame(self, game_state: CMGameState, title: Optional[str] = None):
        """Add a frame to the animation."""
        self.frames.append({
            'game_state': game_state.copy() if hasattr(game_state, 'copy') else game_state,
            'title': title
        })

    def create_animation(self, interval: int = 500, repeat: bool = True) -> FuncAnimation:
        """
        Create an animation from the recorded frames.

        Args:
            interval: Delay between frames in milliseconds
            repeat: Whether to repeat the animation

        Returns:
            matplotlib FuncAnimation object
        """
        if not self.frames:
            raise ValueError("No frames added to animation")

        fig, ax = plt.subplots(figsize=self.figsize)

        def animate(frame_idx):
            ax.clear()
            frame_data = self.frames[frame_idx]

            # Use base renderer to draw the frame
            renderer = MatplotlibRenderer(self.grid_size, self.figsize)
            renderer.fig = fig
            renderer.ax = ax

            renderer._setup_plot()
            renderer._draw_grid()
            renderer._draw_goal(frame_data['game_state'].goal)
            renderer._draw_box(frame_data['game_state'].box)
            renderer._draw_agents(frame_data['game_state'].agents)

            if frame_data['title']:
                ax.set_title(frame_data['title'], fontsize=14)
            else:
                distance = np.linalg.norm(
                    np.array(frame_data['game_state'].box.get_center()) -
                    np.array(frame_data['game_state'].goal.get_center())
                )
                ax.set_title(f'Step: {frame_idx} | Distance: {distance:.2f}', fontsize=14)

            renderer._add_legend(frame_data['game_state'])

            return ax.patches + ax.texts + ax.lines

        anim = FuncAnimation(fig, animate, frames=len(self.frames),
                           interval=interval, blit=False, repeat=repeat)

        return anim

    def save_animation(self, filename: str, interval: int = 500,
                      writer: str = 'pillow', fps: int = 2):
        """
        Save animation to file.

        Args:
            filename: Output filename
            interval: Delay between frames
            writer: Animation writer ('pillow', 'imagemagick', etc.)
            fps: Frames per second for saved animation
        """
        anim = self.create_animation(interval=interval)

        if filename.endswith('.gif'):
            anim.save(filename, writer=writer, fps=fps)
        elif filename.endswith('.mp4'):
            anim.save(filename, writer='ffmpeg', fps=fps)
        else:
            # Default to GIF
            filename += '.gif'
            anim.save(filename, writer=writer, fps=fps)

    def clear_frames(self):
        """Clear all recorded frames."""
        self.frames = []


class TextRenderer:
    """Simple text-based renderer for the CM environment."""

    def __init__(self, grid_size: int = 7):
        """Initialize the text renderer."""
        self.grid_size = grid_size
        self.symbols = {
            'empty': '.',
            'box': 'B',
            'goal': 'G',
            'agent': 'A',
            'box_and_agent': 'B',
            'goal_and_agent': 'G'
        }

    def render(self, game_state: CMGameState) -> str:
        """
        Render the environment as text.

        Args:
            game_state: Current game state

        Returns:
            String representation of the environment
        """
        # Create grid
        grid = [[self.symbols['empty'] for _ in range(self.grid_size)]
                for _ in range(self.grid_size)]

        # Draw goal
        for dx in range(game_state.goal.size):
            for dy in range(game_state.goal.size):
                x, y = game_state.goal.position.x + dx, game_state.goal.position.y + dy
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    grid[x][y] = self.symbols['goal']

        # Draw box
        for dx in range(game_state.box.size):
            for dy in range(game_state.box.size):
                x, y = game_state.box.position.x + dx, game_state.box.position.y + dy
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    grid[x][y] = self.symbols['box']

        # Draw agents
        agent_num = 1
        for agent_id, agent in game_state.agents.items():
            x, y = agent.position.x, agent.position.y
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                grid[x][y] = str(agent_num)
            agent_num += 1

        # Convert to string
        lines = []
        lines.append(f"Step: {game_state.current_step}")
        lines.append(f"Distance to goal: {np.linalg.norm(np.array(game_state.box.get_center()) - np.array(game_state.goal.get_center())):.2f}")
        lines.append(f"Pushing agents: {len(game_state.get_pushing_agents())}")
        lines.append("")

        for row in grid:
            lines.append(' '.join(row))

        return '\n'.join(lines)


def create_visualization_sequence(env, num_steps: int = 20,
                                output_dir: str = "cm_frames"):
    """
    Create a sequence of visualizations for an environment episode.

    Args:
        env: CM environment instance
        num_steps: Number of steps to record
        output_dir: Directory to save frames
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Reset environment
    obs, _ = env.reset()

    renderer = MatplotlibRenderer(env.config.grid_size)

    for step in range(num_steps):
        # Save current state
        renderer.save_frame(
            f"{output_dir}/frame_{step:03d}.png",
            env.game_state,
            f"Step {step}: Distance = {env._get_info()['distance_to_goal']:.2f}"
        )

        # Take random action
        actions = {}
        for agent_id in env.agent_ids:
            avail_actions = env.get_avail_actions(agent_id)
            actions[agent_id] = np.random.choice(avail_actions)

        obs, rewards, terminated, truncated, info = env.step(actions)

        if terminated or truncated:
            # Save final state
            renderer.save_frame(
                f"{output_dir}/frame_{step:03d}_final.png",
                env.game_state,
                f"Step {step}: {'COMPLETED' if terminated else 'TRUNCATED'}"
            )
            break

    print(f"Saved {step + 1} frames to {output_dir}/")
    renderer.close()


def demo_visualizations():
    """Demonstrate different visualization capabilities."""
    from .env_cm import create_cm_env

    print("Creating demonstration visualizations...")

    # Create environment
    env = create_cm_env(difficulty="easy")
    obs, _ = env.reset()

    # Text rendering
    text_renderer = TextRenderer(env.config.grid_size)
    print("\nText Rendering:")
    print(text_renderer.render(env.game_state))

    # Matplotlib rendering
    matplotlib_renderer = MatplotlibRenderer(env.config.grid_size)
    fig = matplotlib_renderer.render_frame(env.game_state, "Demo Frame")
    matplotlib_renderer.save_frame("demo_frame.png", env.game_state, "Demo Visualization")
    plt.show()

    # Create sequence
    create_visualization_sequence(env, num_steps=10, output_dir="demo_frames")

    env.close()
    matplotlib_renderer.close()

    print("Demo visualizations completed!")


if __name__ == "__main__":
    demo_visualizations()