"""
HW3-2: Enhanced DQN Variants (Player Mode) — 40%
==================================================
Implements and compares:
  1. Double DQN   — decouples action selection from Q-value evaluation
  2. Dueling DQN  — separate value (V) and advantage (A) streams

Environment: GridWorld in 'player' mode
  - Goal/Pit/Wall are fixed; Player starts at a random free cell.

Why Double DQN?
  Standard DQN overestimates Q-values because it uses the same network
  for both selecting and evaluating actions. Double DQN fixes this:
    y = r + γ · Q_target(s', argmax_a Q_online(s', a))

Why Dueling DQN?
  Many states have similar Q-values across actions. Dueling DQN separates
  Q(s,a) = V(s) + A(s,a) - mean_a'[A(s,a')]
  making V(s) easier to learn when actions have little effect on outcome.
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from environment import GridWorld

# ── Hyper-parameters ──────────────────────────────────────────────────────────
STATE_SIZE   = 12
ACTION_SIZE  = 4
LR           = 1e-3
GAMMA        = 0.99
EPSILON      = 1.0
EPSILON_MIN  = 0.01
EPSILON_DECAY= 0.995
BUFFER_SIZE  = 10_000
BATCH_SIZE   = 64
EPISODES     = 600
MAX_STEPS    = 50
UPDATE_EVERY = 4          # train every N steps
TARGET_UPDATE= 20         # sync target network every N episodes

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


# ── Replay Buffer ─────────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *transition):
        self.buffer.append(transition)

    def sample(self, n):
        batch = random.sample(self.buffer, n)
        s, a, r, s2, d = zip(*batch)
        return (
            torch.tensor(np.array(s),  dtype=torch.float32),
            torch.tensor(a,            dtype=torch.long),
            torch.tensor(r,            dtype=torch.float32),
            torch.tensor(np.array(s2), dtype=torch.float32),
            torch.tensor(d,            dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ── Networks ───────────────────────────────────────────────────────────────────
class StandardQNet(nn.Module):
    """Standard Q-network (used for both Vanilla DQN baseline & Double DQN)."""
    def __init__(self, state_size, action_size, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),     nn.ReLU(),
            nn.Linear(hidden, action_size),
        )

    def forward(self, x):
        return self.net(x)


class DuelingQNet(nn.Module):
    """
    Dueling Q-network.

    Shared body  →  Value stream  V(s)
                 →  Advantage stream A(s,a)
    Q(s,a) = V(s) + A(s,a) − mean_a[A(s,a)]
    """
    def __init__(self, state_size, action_size, hidden=64):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_size, hidden), nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_size),
        )

    def forward(self, x):
        # Support both 1-D (single state) and 2-D (batch) inputs
        single = x.dim() == 1
        if single:
            x = x.unsqueeze(0)
        feat = self.feature(x)
        v    = self.value_stream(feat)
        a    = self.advantage_stream(feat)
        # Combine: subtract mean advantage for identifiability
        q = v + a - a.mean(dim=1, keepdim=True)
        return q.squeeze(0) if single else q


# ── Generic Agent ─────────────────────────────────────────────────────────────
class DQNVariantAgent:
    """
    Supports three DQN variants:
      'standard'  — original DQN (used as baseline)
      'double'    — Double DQN
      'dueling'   — Dueling DQN (with Double DQN target computation)
    """
    def __init__(self, variant='standard'):
        self.variant = variant

        if variant in ('standard', 'double'):
            self.online_net = StandardQNet(STATE_SIZE, ACTION_SIZE)
            self.target_net = StandardQNet(STATE_SIZE, ACTION_SIZE)
        elif variant == 'dueling':
            self.online_net = DuelingQNet(STATE_SIZE, ACTION_SIZE)
            self.target_net = DuelingQNet(STATE_SIZE, ACTION_SIZE)
        else:
            raise ValueError(f"Unknown variant: {variant}")

        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=LR)
        self.buffer    = ReplayBuffer(BUFFER_SIZE)
        self.epsilon   = EPSILON
        self.step_cnt  = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(ACTION_SIZE)
        with torch.no_grad():
            return self.online_net(
                torch.tensor(state, dtype=torch.float32)
            ).argmax().item()

    def push(self, *t):
        self.buffer.push(*t)

    def learn(self):
        if len(self.buffer) < BATCH_SIZE:
            return
        self.step_cnt += 1
        if self.step_cnt % UPDATE_EVERY != 0:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)

        # Current Q-values from online network
        q_current = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.variant == 'standard':
                # Classic DQN target
                q_next = self.target_net(next_states).max(1)[0]
            else:
                # Double DQN target (used for both 'double' and 'dueling')
                best_actions = self.online_net(next_states).argmax(1)
                q_next = self.target_net(next_states).gather(
                    1, best_actions.unsqueeze(1)).squeeze(1)

            q_target = rewards + GAMMA * q_next * (1 - dones)

        loss = nn.functional.mse_loss(q_current, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

    def sync_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())


# ── Training function ─────────────────────────────────────────────────────────
def train_variant(variant: str, episodes: int = EPISODES):
    env   = GridWorld(mode='player')
    agent = DQNVariantAgent(variant=variant)

    rewards_log = []
    success_log = []

    for ep in range(1, episodes + 1):
        state     = env.reset()
        ep_reward = 0

        for _ in range(MAX_STEPS):
            action              = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.push(state, action, reward, next_state, float(done))
            agent.learn()
            state               = next_state
            ep_reward          += reward
            if done:
                break

        if ep % TARGET_UPDATE == 0:
            agent.sync_target()

        rewards_log.append(ep_reward)
        success_log.append(1 if ep_reward > 0 else 0)

        if ep % 100 == 0:
            avg_r = np.mean(rewards_log[-50:])
            win   = np.mean(success_log[-50:]) * 100
            print(f"[{variant.upper():9s} / Player] Ep {ep:4d} | "
                  f"AvgReward={avg_r:6.2f} | WinRate={win:5.1f}% | ε={agent.epsilon:.3f}")

    return np.array(rewards_log), np.array(success_log)


# ── Compare all variants ───────────────────────────────────────────────────────
def compare():
    print("=" * 60)
    print("Training Standard DQN (baseline) ...")
    std_r, std_s = train_variant('standard')

    print("=" * 60)
    print("Training Double DQN ...")
    dbl_r, dbl_s = train_variant('double')

    print("=" * 60)
    print("Training Dueling DQN ...")
    duel_r, duel_s = train_variant('dueling')

    # ── Plot ──────────────────────────────────────────────────────────────────
    window = 30

    def smooth(arr):
        return np.convolve(arr, np.ones(window) / window, mode='valid')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Reward curves
    axes[0].plot(smooth(std_r),  label='Standard DQN',  color='steelblue')
    axes[0].plot(smooth(dbl_r),  label='Double DQN',    color='darkorange')
    axes[0].plot(smooth(duel_r), label='Dueling DQN',   color='seagreen')
    axes[0].set_title("DQN Variants — Player Mode\nSmoothed Episode Reward")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total Reward")
    axes[0].legend()

    # Win-rate curves
    axes[1].plot(smooth(std_s) * 100,  label='Standard DQN',  color='steelblue')
    axes[1].plot(smooth(dbl_s) * 100,  label='Double DQN',    color='darkorange')
    axes[1].plot(smooth(duel_s) * 100, label='Dueling DQN',   color='seagreen')
    axes[1].set_title("DQN Variants — Player Mode\nWin Rate (%)")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Win Rate (%)")
    axes[1].set_ylim(0, 105)
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "hw3_2_dqn_variants_player.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Plot saved → {path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n{'='*60}")
    print(f"{'Variant':<15} {'Final Avg Reward':>18} {'Final Win Rate':>15}")
    print("-" * 50)
    for name, r, s in [('Standard DQN', std_r, std_s),
                        ('Double DQN',   dbl_r, dbl_s),
                        ('Dueling DQN',  duel_r, duel_s)]:
        avg_r = np.mean(r[-50:])
        win   = np.mean(s[-50:]) * 100
        print(f"{name:<15} {avg_r:>18.2f} {win:>14.1f}%")
    print("=" * 50)


if __name__ == '__main__':
    compare()
