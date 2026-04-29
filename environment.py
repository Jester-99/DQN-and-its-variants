"""
GridWorld Environment for DQN Homework 3
=========================================
Classic 3×4 GridWorld with three modes:
  - static: Player at (2,0), Goal at (0,3), Pit at (1,3), Wall at (1,1)
  - player: Player random, Goal/Pit/Wall fixed as above
  - random: All positions random

Grid layout (3 rows × 4 cols):
  (0,0) (0,1) (0,2) (0,3)=Goal
  (1,0) (1,1)=Wall (1,2) (1,3)=Pit
  (2,0)=Player (2,1) (2,2) (2,3)

Actions: 0=Up, 1=Down, 2=Left, 3=Right

Optimal path (static mode):
  (2,0)→(1,0)→(0,0)→(0,1)→(0,2)→(0,3)=Goal  (reward = 5·(-1) + 10 = +5)
"""

import numpy as np
import random

ACTIONS = ['up', 'down', 'left', 'right']
ACTION_MAP = {
    0: (-1, 0),   # up
    1: (1, 0),    # down
    2: (0, -1),   # left
    3: (0, 1),    # right
}

class GridWorld:
    """
    A 3×4 GridWorld environment with static, player, and random modes.

    Cell types:
      'G' = Goal   (reward +10, terminal)
      'P' = Pit    (reward -10, terminal)
      'W' = Wall   (blocked)
      '+' = Player
      '-' = Empty
    """

    def __init__(self, mode='static', size=(3, 4)):
        self.mode = mode
        self.rows, self.cols = size
        self.reset()

    def _init_board(self):
        """Initialize the grid based on mode."""
        if self.mode == 'static':
            self.player_pos = (2, 0)
            self.goal_pos   = (0, 3)
            self.pit_pos    = (1, 3)
            self.wall_pos   = (1, 1)
        elif self.mode == 'player':
            # Fixed special cells, random player
            self.goal_pos = (0, 3)
            self.pit_pos  = (1, 3)
            self.wall_pos = (1, 1)
            special = {self.goal_pos, self.pit_pos, self.wall_pos}
            all_cells = [(r, c) for r in range(self.rows)
                         for c in range(self.cols)]
            free = [cell for cell in all_cells if cell not in special]
            self.player_pos = random.choice(free)
        elif self.mode == 'random':
            all_cells = [(r, c) for r in range(self.rows)
                         for c in range(self.cols)]
            chosen = random.sample(all_cells, 4)
            self.player_pos = chosen[0]
            self.goal_pos   = chosen[1]
            self.pit_pos    = chosen[2]
            self.wall_pos   = chosen[3]
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def reset(self):
        """Reset the environment and return the initial state vector."""
        self._init_board()
        self.done = False
        return self._get_state()

    def _get_state(self):
        """
        State vector of length rows*cols = 12.
        Encoding: -1=Wall, -0.5=Pit, +0.5=Goal, +1=Player, 0=Empty
        """
        state = np.zeros(self.rows * self.cols, dtype=np.float32)
        idx = lambda r, c: r * self.cols + c
        state[idx(*self.goal_pos)]   = 0.5
        state[idx(*self.pit_pos)]    = -0.5
        state[idx(*self.wall_pos)]   = -1.0
        state[idx(*self.player_pos)] = 1.0
        return state

    def step(self, action):
        """
        Take an action and return (next_state, reward, done).
        action: int in [0,1,2,3]
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset().")

        dr, dc = ACTION_MAP[action]
        nr = self.player_pos[0] + dr
        nc = self.player_pos[1] + dc

        # Boundary / wall check — stay in place if invalid
        if 0 <= nr < self.rows and 0 <= nc < self.cols \
                and (nr, nc) != self.wall_pos:
            self.player_pos = (nr, nc)

        # Compute reward & done
        if self.player_pos == self.goal_pos:
            reward = 10
            self.done = True
        elif self.player_pos == self.pit_pos:
            reward = -10
            self.done = True
        else:
            reward = -1  # step penalty

        return self._get_state(), reward, self.done

    def render(self):
        """Print a text representation of the grid."""
        symbols = {
            self.goal_pos:   'G',
            self.pit_pos:    'P',
            self.wall_pos:   'W',
            self.player_pos: '+',
        }
        for r in range(self.rows):
            row_str = ''
            for c in range(self.cols):
                row_str += symbols.get((r, c), '-') + ' '
            print(row_str)
        print()

    @property
    def state_size(self):
        return self.rows * self.cols   # 12

    @property
    def action_size(self):
        return len(ACTIONS)            # 4
