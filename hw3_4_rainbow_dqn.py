"""
HW3-4: Rainbow DQN (Bonus) — Random Mode
=========================================
Rainbow DQN combines several improvements to the basic DQN:
  1. Double DQN (Decoupled selection/evaluation)
  2. Dueling DQN (Value/Advantage separation)
  3. Prioritized Experience Replay (PER)
  4. Multi-step Learning (N-step returns)
  5. Noisy Networks (Parametric noise for exploration)

This implementation integrates these components into a single agent
to solve the Random Mode GridWorld.
"""

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from environment import GridWorld

# ── Hyper-parameters ──────────────────────────────────────────────────────────
STATE_SIZE    = 12
ACTION_SIZE   = 4
LR            = 3e-4
GAMMA         = 0.99
N_STEPS       = 3            # N-step learning
BUFFER_SIZE   = 20_000
BATCH_SIZE    = 128
EPISODES      = 1000
MAX_STEPS     = 100
TAU           = 0.005        # Soft update
ALPHA_PER     = 0.6
BETA_PER      = 0.4
PER_EPS       = 1e-5

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Noisy Linear Layer ────────────────────────────────────────────────────────
class NoisyLinear(nn.Module):
    """
    Noisy Linear Layer for exploration (Fortunato et al., 2017).
    Replaces epsilon-greedy exploration with parametric noise in weights.
    """
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        if self.training:
            return F.linear(x, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(x, self.weight_mu, self.bias_mu)

# ── Rainbow Network ───────────────────────────────────────────────────────────
class RainbowNet(nn.Module):
    """Dueling architecture with Noisy Linear layers."""
    def __init__(self, state_size=STATE_SIZE, action_size=ACTION_SIZE, hidden=128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_size, hidden), nn.ReLU(),
        )
        # Value stream
        self.value_stream = nn.Sequential(
            NoisyLinear(hidden, hidden), nn.ReLU(),
            NoisyLinear(hidden, 1)
        )
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            NoisyLinear(hidden, hidden), nn.ReLU(),
            NoisyLinear(hidden, action_size)
        )

    def forward(self, x):
        feat = self.feature(x)
        v = self.value_stream(feat)
        a = self.advantage_stream(feat)
        return v + a - a.mean(dim=-1, keepdim=True)

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

# ── Prioritized Replay Buffer with N-step Support ─────────────────────────────
class RainbowBuffer:
    def __init__(self, capacity, n_step=N_STEPS, gamma=GAMMA):
        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.n_step_buffer = deque(maxlen=n_step)

    def push(self, s, a, r, s2, d):
        # 1. Add to n-step buffer
        self.n_step_buffer.append((s, a, r, s2, d))
        if len(self.n_step_buffer) < self.n_step:
            return

        # 2. Compute n-step reward and state
        # r_n = r0 + g*r1 + g^2*r2 ...
        reward, next_state, done = self._get_n_step_info()
        state, action = self.n_step_buffer[0][:2]

        # 3. Store in main buffer
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def _get_n_step_info(self):
        reward = 0
        for i, transition in enumerate(self.n_step_buffer):
            reward += (self.gamma ** i) * transition[2]
            if transition[4]: # done
                return reward, transition[3], transition[4]
        return reward, self.n_step_buffer[-1][3], self.n_step_buffer[-1][4]

    def sample(self, n, beta=BETA_PER):
        size = len(self.buffer)
        probs = self.priorities[:size] ** ALPHA_PER
        probs /= probs.sum()

        indices = np.random.choice(size, n, p=probs)
        samples = [self.buffer[i] for i in indices]

        weights = (size * probs[indices]) ** (-beta)
        weights /= weights.max()

        s, a, r, s2, d = zip(*samples)
        return (
            torch.tensor(np.array(s), dtype=torch.float32),
            torch.tensor(a,           dtype=torch.long),
            torch.tensor(r,           dtype=torch.float32),
            torch.tensor(np.array(s2),dtype=torch.float32),
            torch.tensor(d,           dtype=torch.float32),
            torch.tensor(weights,     dtype=torch.float32),
            indices
        )

    def update_priorities(self, indices, errors):
        for idx, err in zip(indices, errors):
            self.priorities[idx] = abs(err) + PER_EPS

    def __len__(self):
        return len(self.buffer)

from collections import deque

# ── Rainbow Agent ─────────────────────────────────────────────────────────────
class RainbowAgent:
    def __init__(self):
        self.online_net = RainbowNet()
        self.target_net = RainbowNet()
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=LR)
        self.buffer = RainbowBuffer(BUFFER_SIZE)
        
    def select_action(self, state):
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            return self.online_net(state_t).argmax().item()

    def learn(self):
        if len(self.buffer) < BATCH_SIZE:
            return
        
        s, a, r, s2, d, w, idx = self.buffer.sample(BATCH_SIZE)
        
        q_curr = self.online_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            # Double DQN
            best_a = self.online_net(s2).argmax(1)
            q_next = self.target_net(s2).gather(1, best_a.unsqueeze(1)).squeeze(1)
            # Target with n-step gamma
            q_target = r + (GAMMA ** N_STEPS) * q_next * (1 - d)
            
        loss = (w * F.mse_loss(q_curr, q_target, reduction='none')).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Polyak update
        for tp, op in zip(self.target_net.parameters(), self.online_net.parameters()):
            tp.data.copy_(TAU * op.data + (1 - TAU) * tp.data)
            
        self.online_net.reset_noise()
        self.target_net.reset_noise()
        
        # Update PER priorities
        errors = (q_curr - q_target).detach().abs().cpu().numpy()
        self.buffer.update_priorities(idx, errors)

# ── Training ──────────────────────────────────────────────────────────────────
def train():
    env = GridWorld(mode='random')
    agent = RainbowAgent()
    
    rewards_log = []
    success_log = []
    
    for ep in range(1, EPISODES + 1):
        state = env.reset()
        ep_reward = 0
        for _ in range(MAX_STEPS):
            action = agent.select_action(state)
            s2, r, done = env.step(action)
            agent.buffer.push(state, action, r, s2, done)
            agent.learn()
            state = s2
            ep_reward += r
            if done: break
            
        rewards_log.append(ep_reward)
        success_log.append(1 if ep_reward > 0 else 0)
        
        if ep % 100 == 0:
            avg_r = np.mean(rewards_log[-50:])
            win = np.mean(success_log[-50:]) * 100
            print(f"[Rainbow DQN / Random] Ep {ep:4d} | AvgR={avg_r:6.2f} | Win={win:5.1f}%")
            
    # Plot
    window = 40
    def smooth(arr): return np.convolve(arr, np.ones(window)/window, mode='valid')
    
    plt.figure(figsize=(10, 5))
    plt.plot(smooth(rewards_log), color='indigo', label='Rainbow DQN')
    plt.title("Rainbow DQN — Random Mode Performance")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    path = os.path.join(SAVE_DIR, "hw3_4_rainbow_random.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Plot saved → {path}")
    
    return rewards_log

if __name__ == "__main__":
    train()
