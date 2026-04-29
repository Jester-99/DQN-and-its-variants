"""
HW3-1: Naive DQN (Static Mode) — 30%
=======================================
Implements a basic DQN with an Experience Replay Buffer.

Environment: GridWorld in 'static' mode
  - Player always starts at (0,3)
  - Goal at (0,0), Pit at (0,1), Wall at (1,1)

Architecture:
  Input  → Linear(state_size, 64) → ReLU
         → Linear(64, 64)         → ReLU
         → Linear(64, action_size)

Training loop:
  1. ε-greedy action selection
  2. Store (s, a, r, s', done) in replay buffer
  3. Sample random mini-batch
  4. Compute Q-targets using Bellman equation
  5. Update Q-network with MSE loss
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — no GUI window needed
import matplotlib.pyplot as plt
import os

from environment import GridWorld

# ── Hyper-parameters ──────────────────────────────────────────────────────────
STATE_SIZE   = 12     # 3×4 grid flattened
ACTION_SIZE  = 4
LR           = 1e-3
GAMMA        = 0.99
EPSILON      = 1.0
EPSILON_MIN  = 0.01
EPSILON_DECAY= 0.995
BUFFER_SIZE  = 10_000
BATCH_SIZE   = 64
EPISODES     = 500
MAX_STEPS    = 50

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Replay Buffer ─────────────────────────────────────────────────────────────
class ReplayBuffer:
    """
    Experience Replay Buffer.

    Stores transitions (state, action, reward, next_state, done) and
    supports uniform random sampling of mini-batches.

    Why it helps:
      • Breaks temporal correlations between consecutive transitions,
        stabilizing gradient updates.
      • Allows each experience to be replayed multiple times,
        improving data efficiency.
    """
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states),      dtype=torch.float32),
            torch.tensor(actions,                dtype=torch.long),
            torch.tensor(rewards,                dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones,                  dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ── Q-Network ─────────────────────────────────────────────────────────────────
class QNetwork(nn.Module):
    """
    Simple 3-layer fully-connected Q-network.

    Input  : state vector (length = state_size)
    Output : Q-values for each action (length = action_size)
    """
    def __init__(self, state_size: int, action_size: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_size),
        )

    def forward(self, x):
        return self.net(x)


# ── DQN Agent ─────────────────────────────────────────────────────────────────
class DQNAgent:
    """
    Naive DQN agent with Experience Replay.

    Key DQN idea (Mnih et al., 2013):
      Q(s,a) ← Q(s,a) + α [r + γ·max_a' Q(s',a') − Q(s,a)]

    Using a replay buffer allows us to compute these TD-updates on
    decorrelated mini-batches rather than single correlated transitions.
    """
    def __init__(self):
        self.q_net    = QNetwork(STATE_SIZE, ACTION_SIZE)
        self.optimizer= optim.Adam(self.q_net.parameters(), lr=LR)
        self.buffer   = ReplayBuffer(BUFFER_SIZE)
        self.epsilon  = EPSILON

    def select_action(self, state):
        """ε-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randrange(ACTION_SIZE)
        with torch.no_grad():
            q_vals = self.q_net(torch.tensor(state, dtype=torch.float32))
        return q_vals.argmax().item()

    def learn(self):
        if len(self.buffer) < BATCH_SIZE:
            return
        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)

        # Current Q-values
        q_current = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values (standard DQN)
        with torch.no_grad():
            q_next = self.q_net(next_states).max(1)[0]
            q_target = rewards + GAMMA * q_next * (1 - dones)

        loss = nn.functional.mse_loss(q_current, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

    def push(self, *args):
        self.buffer.push(*args)


# ── Training ──────────────────────────────────────────────────────────────────
def train():
    env   = GridWorld(mode='static')
    agent = DQNAgent()

    total_rewards = []
    successes     = []

    for ep in range(1, EPISODES + 1):
        state = env.reset()
        ep_reward = 0

        for _ in range(MAX_STEPS):
            action     = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.push(state, action, reward, next_state, float(done))
            agent.learn()
            state      = next_state
            ep_reward += reward
            if done:
                break

        total_rewards.append(ep_reward)
        successes.append(1 if ep_reward > 0 else 0)

        if ep % 50 == 0:
            avg_r  = np.mean(total_rewards[-50:])
            win_r  = np.mean(successes[-50:]) * 100
            print(f"[Naive DQN / Static] Ep {ep:4d} | "
                  f"AvgReward={avg_r:6.2f} | WinRate={win_r:5.1f}% | ε={agent.epsilon:.3f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    window = 20
    smoothed = np.convolve(total_rewards,
                           np.ones(window) / window, mode='valid')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(smoothed, color='steelblue')
    axes[0].set_title("Naive DQN — Static Mode\nSmoothed Episode Reward")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total Reward")

    win_smooth = np.convolve(successes,
                             np.ones(window) / window, mode='valid') * 100
    axes[1].plot(win_smooth, color='darkorange')
    axes[1].set_title("Naive DQN — Static Mode\nWin Rate (%)")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Win Rate (%)")
    axes[1].set_ylim(0, 105)

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "hw3_1_naive_dqn_static.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Plot saved → {path}")

    return agent, total_rewards


if __name__ == '__main__':
    agent, rewards = train()
