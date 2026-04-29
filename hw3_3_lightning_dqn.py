"""
HW3-3: Enhanced DQN for Random Mode with Training Tips — 30%
==============================================================
Converts the DQN model to PyTorch Lightning and adds stabilization techniques.

Training Tips Implemented:
  1. Gradient Clipping        — clips gradients to max-norm 1.0
  2. Cosine Annealing LR      — LR smoothly annealed from LR to LR_MIN
  3. Prioritized Experience   — high-TD-error transitions sampled more
     Replay (PER)
  4. Soft Target Update       — Polyak averaging (τ=0.005)

Framework: PyTorch Lightning (LightningModule + Trainer)
  Epochs map 1:1 to episodes; the module collects ONE rollout per epoch in
  on_train_epoch_start, then the Trainer performs ONE gradient step per epoch.

Architecture: Dueling Double DQN
"""

import numpy as np
import random
from collections import deque
import os

import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L
from torch.utils.data import IterableDataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from environment import GridWorld

# ── Hyper-parameters ──────────────────────────────────────────────────────────
STATE_SIZE    = 12
ACTION_SIZE   = 4
LR            = 3e-4
LR_MIN        = 1e-5
GAMMA         = 0.99
EPSILON_START = 1.0
EPSILON_MIN   = 0.02
EPSILON_DECAY = 0.997
BUFFER_SIZE   = 20_000
BATCH_SIZE    = 128
EPISODES      = 1_000
MAX_STEPS     = 100
TAU           = 0.005        # Polyak soft-update coefficient
GRAD_CLIP     = 1.0          # gradient clip max-norm
ALPHA_PER     = 0.6          # PER priority exponent
BETA_START    = 0.4          # IS correction exponent (anneals to 1)
PER_EPS       = 1e-5         # min priority to avoid zero sampling

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


# ── Prioritized Replay Buffer ─────────────────────────────────────────────────
class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (PER) — Schaul et al. 2016.

    Transitions with larger |TD-error| are sampled more frequently,
    helping the agent focus on the most informative experiences.
    Importance-sampling weights correct for the induced bias.
    """
    def __init__(self, capacity=BUFFER_SIZE, alpha=ALPHA_PER):
        self.capacity = capacity
        self.alpha    = alpha
        self.buffer   = []
        self.prios    = np.zeros(capacity, dtype=np.float32)
        self.pos      = 0

    def push(self, s, a, r, s2, d):
        max_p = self.prios.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((s, a, r, s2, d))
        else:
            self.buffer[self.pos] = (s, a, r, s2, d)
        self.prios[self.pos] = max_p
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, n, beta=BETA_START):
        size   = len(self.buffer)
        prios  = self.prios[:size]
        probs  = prios ** self.alpha
        probs /= probs.sum()

        replace = n > size
        indices = np.random.choice(size, n, p=probs, replace=replace)
        samples = [self.buffer[i] for i in indices]

        weights = (size * probs[indices]) ** (-beta)
        weights /= weights.max()

        s, a, r, s2, d = zip(*samples)
        return (
            torch.tensor(np.array(s),  dtype=torch.float32),
            torch.tensor(a,            dtype=torch.long),
            torch.tensor(r,            dtype=torch.float32),
            torch.tensor(np.array(s2), dtype=torch.float32),
            torch.tensor(d,            dtype=torch.float32),
            torch.tensor(weights,      dtype=torch.float32),
            indices,
        )

    def update_priorities(self, indices, td_errors):
        for i, e in zip(indices, td_errors):
            self.prios[i] = abs(float(e)) + PER_EPS

    def __len__(self):
        return len(self.buffer)


# ── Dueling Q-Network ─────────────────────────────────────────────────────────
class DuelingQNet(nn.Module):
    """Dueling DQN: Q(s,a) = V(s) + A(s,a) − mean_a[A(s,a)]"""
    def __init__(self, state_size=STATE_SIZE, action_size=ACTION_SIZE, hidden=128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_size, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),     nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(),
            nn.Linear(64, action_size),
        )

    def forward(self, x):
        single = x.dim() == 1
        if single:
            x = x.unsqueeze(0)
        feat = self.feature(x)
        v    = self.value_stream(feat)
        a    = self.advantage_stream(feat)
        q    = v + a - a.mean(dim=1, keepdim=True)
        return q.squeeze(0) if single else q


# ── Single-batch IterableDataset ──────────────────────────────────────────────
class SingleBatchDataset(IterableDataset):
    """Provides exactly one batch per epoch (Lightning step = one gradient step)."""
    def __init__(self, batch):
        self.batch = batch

    def __iter__(self):
        yield self.batch


# ── PyTorch Lightning Module ───────────────────────────────────────────────────
class DQNLightning(L.LightningModule):
    """
    Dueling Double DQN as a LightningModule.

    One Lightning epoch = one DRL episode:
      1. on_train_epoch_start: collect one rollout (pushes to PER buffer)
      2. training_step:        one gradient update on a mini-batch
      3. After step:           soft Polyak target update

    Training Tips integrated:
      • Gradient clipping (set via Trainer(gradient_clip_val=…))
      • Cosine Annealing LR (configure_optimizers)
      • PER (PrioritizedReplayBuffer + weighted MSE loss)
      • Soft target update (_soft_update called in training_step)
    """
    def __init__(self, buffer: PrioritizedReplayBuffer):
        super().__init__()
        self.buffer      = buffer
        self.online_net  = DuelingQNet()
        self.target_net  = DuelingQNet()
        self.target_net.load_state_dict(self.online_net.state_dict())
        for p in self.target_net.parameters():
            p.requires_grad_(False)

        self.epsilon     = EPSILON_START
        self.rewards_log = []
        self.success_log = []
        self._env        = GridWorld(mode='random')
        self._batch      = None   # filled by on_train_epoch_start

    # ── Environment interaction (called before gradient step each epoch) ───────
    def on_train_epoch_start(self):
        ep_reward = self._collect_episode()
        self.rewards_log.append(ep_reward)
        self.success_log.append(1 if ep_reward > 0 else 0)

        # Sample mini-batch from PER
        ep = self.current_epoch + 1
        beta = min(1.0, BETA_START + ep / EPISODES * (1.0 - BETA_START))
        if len(self.buffer) >= BATCH_SIZE:
            self._batch = self.buffer.sample(BATCH_SIZE, beta=beta)
        else:
            self._batch = None

    def _collect_episode(self) -> float:
        """Run one ε-greedy episode, push transitions to PER buffer."""
        state     = self._env.reset()
        ep_reward = 0.0
        for _ in range(MAX_STEPS):
            if random.random() < self.epsilon:
                action = random.randrange(ACTION_SIZE)
            else:
                with torch.no_grad():
                    action = self.online_net(
                        torch.tensor(state, dtype=torch.float32)
                    ).argmax().item()

            next_state, reward, done = self._env.step(action)
            self.buffer.push(state, action, reward, next_state, float(done))
            state      = next_state
            ep_reward += reward
            if done:
                break

        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
        return ep_reward

    # ── Gradient update ────────────────────────────────────────────────────────
    def training_step(self, batch, batch_idx):
        if self._batch is None:
            # Not enough data yet; return zero loss
            return torch.tensor(0.0, requires_grad=True)

        states, actions, rewards, next_states, dones, weights, indices = self._batch

        # Move to device
        states, actions, rewards = states.to(self.device), \
                                   actions.to(self.device), \
                                   rewards.to(self.device)
        next_states, dones       = next_states.to(self.device), \
                                   dones.to(self.device)
        weights                  = weights.to(self.device)

        # Online Q-values
        q_pred = self.online_net(states).gather(
            1, actions.unsqueeze(1)).squeeze(1)

        # Double-DQN target
        with torch.no_grad():
            best_a  = self.online_net(next_states).argmax(1)
            q_next  = self.target_net(next_states).gather(
                1, best_a.unsqueeze(1)).squeeze(1)
            q_target = rewards + GAMMA * q_next * (1.0 - dones)

        td_errors = (q_pred - q_target).detach().cpu().numpy()
        self.buffer.update_priorities(indices, td_errors)

        # Weighted MSE (PER importance-sampling correction)
        loss = (weights * nn.functional.mse_loss(
            q_pred, q_target, reduction='none')).mean()

        self.log('train_loss', loss, prog_bar=False, on_step=False, on_epoch=True)

        # ── Soft target update (Polyak) ────────────────────────────────────────
        with torch.no_grad():
            for tp, op in zip(self.target_net.parameters(),
                              self.online_net.parameters()):
                tp.data.copy_(TAU * op.data + (1.0 - TAU) * tp.data)

        return loss

    # ── Optimizer + LR Scheduler (Tip 2: Cosine Annealing) ────────────────────
    def configure_optimizers(self):
        optimizer = optim.Adam(self.online_net.parameters(), lr=LR)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPISODES, eta_min=LR_MIN)
        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'}}

    # ── DataLoader: one dummy batch per epoch so Lightning calls training_step ─
    def train_dataloader(self):
        # Return a tiny dummy dataset — actual data comes from self._batch
        dummy = [(torch.zeros(1),)]
        return DataLoader(dummy)

    def on_train_epoch_end(self):
        ep = self.current_epoch + 1
        if ep % 100 == 0:
            avg_r = np.mean(self.rewards_log[-50:])
            win   = np.mean(self.success_log[-50:]) * 100
            print(f"[PL Dueling DDQN / Random] Ep {ep:4d} | "
                  f"AvgReward={avg_r:6.2f} | WinRate={win:5.1f}% "
                  f"| ε={self.epsilon:.3f}")


# ── Main ─────────────────────────────────────────────────────────────────────
def train():
    buffer = PrioritizedReplayBuffer()

    # Pre-fill buffer
    print("Pre-filling replay buffer ...")
    env_pre = GridWorld(mode='random')
    while len(buffer) < BATCH_SIZE * 4:
        s = env_pre.reset()
        for _ in range(MAX_STEPS):
            a        = random.randrange(ACTION_SIZE)
            s2, r, d = env_pre.step(a)
            buffer.push(s, a, r, s2, float(d))
            s = s2
            if d:
                break
    print(f"Buffer pre-filled: {len(buffer)} transitions.\n")

    module = DQNLightning(buffer)

    trainer = L.Trainer(
        max_epochs=EPISODES,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        gradient_clip_val=GRAD_CLIP,           # ← Training Tip 1: Gradient Clip
        gradient_clip_algorithm='norm',
    )

    trainer.fit(module)

    # ── Plot ──────────────────────────────────────────────────────────────────
    window = 40

    def smooth(arr):
        return np.convolve(arr, np.ones(window) / window, mode='valid')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(smooth(module.rewards_log), color='mediumpurple')
    axes[0].set_title(
        "PyTorch Lightning Dueling DDQN — Random Mode\n"
        "Smoothed Episode Reward\n"
        "(Grad Clip | CosineAnnealingLR | PER | Soft Target Update)")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total Reward")

    axes[1].plot(smooth(module.success_log) * 100, color='crimson')
    axes[1].set_title("Win Rate (%)")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Win Rate (%)")
    axes[1].set_ylim(0, 105)

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "hw3_3_pl_dqn_random.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\nPlot saved → {path}")

    return module


if __name__ == '__main__':
    model = train()
