# Homework 3: DQN and Its Variants

## Project Structure

```
dqn_hw3/
├── environment.py          # Custom 2×4 GridWorld (static / player / random)
├── hw3_1_naive_dqn.py      # HW3-1: Naive DQN + Experience Replay (static)
├── hw3_1_report.md         # HW3-1: Understanding report
├── hw3_2_enhanced_dqn.py   # HW3-2: Double DQN & Dueling DQN (player)
├── hw3_3_lightning_dqn.py  # HW3-3: PyTorch Lightning DQN + tips (random)
└── README.md               # This file
```

---

## Environment

A 2 × 4 GridWorld with three modes:

| Mode | Player Position | Goal/Pit/Wall Position | Use Case |
|---|---|---|---|
| `static` | Fixed (0,3) | Fixed | Test correctness / reproducibility |
| `player` | Random | Fixed | Test strategy with varying starts |
| `random` | Random | Random | Train a stronger, generalised policy |

**Encoding:** state vector of length 8 (2×4 flattened):
- `-1.0` = Wall  
- `-0.5` = Pit  
- `+0.5` = Goal  
- `+1.0` = Player  
- `0.0`  = Empty

**Rewards:** Goal = +10 (terminal), Pit = −10 (terminal), step = −1.

---

## HW3-1: Naive DQN (Static Mode) — 30%

**Run:**
```bash
python hw3_1_naive_dqn.py
```

**Key concepts:**
- Q-network: 3-layer MLP (8 → 64 → 64 → 4)
- Experience Replay Buffer (capacity 10,000)
- ε-greedy exploration (ε decays 1.0 → 0.01)
- Adam optimizer, MSE loss, γ = 0.99

**Output:** `hw3_1_naive_dqn_static.png`

See `hw3_1_report.md` for the full understanding report.

---

## HW3-2: Enhanced DQN Variants (Player Mode) — 40%

**Run:**
```bash
python hw3_2_enhanced_dqn.py
```

Compares three agents on the **player** mode (random start):

| Variant | Key Idea |
|---|---|
| **Standard DQN** | Baseline; Q-target uses max over target network |
| **Double DQN** | Decouple action selection (online) from evaluation (target) to reduce overestimation |
| **Dueling DQN** | Separate V(s) and A(s,a) streams; Q = V + A − mean(A) |

**Double DQN target:**
```
y = r + γ · Q_target(s', argmax_a Q_online(s', a))
```

**Dueling network output:**
```
Q(s,a) = V(s) + A(s,a) − (1/|A|) Σ_{a'} A(s,a')
```

**Output:** `hw3_2_dqn_variants_player.png` + summary table in terminal.

---

## HW3-3: PyTorch Lightning DQN + Training Tips (Random Mode) — 30%

**Run:**
```bash
python hw3_3_lightning_dqn.py
```

**Framework conversion:** PyTorch → **PyTorch Lightning**

**Training stabilisation techniques:**

| Technique | Benefit |
|---|---|
| **Gradient Clipping** (max-norm 1.0) | Prevents exploding gradients in early training |
| **Cosine Annealing LR** (LR_min = 1e-5) | Smooth LR decay avoids oscillation near optima |
| **Prioritized Experience Replay (PER)** | Focuses replay on high-TD-error transitions |
| **Soft Target Update** (τ = 0.005) | Smoothly tracks online net; more stable than hard copy |

Architecture: **Dueling Double DQN** (best of HW3-2 + tips).

**Output:** `hw3_3_pl_dqn_random.png`

---

## Installation

```bash
pip install torch lightning matplotlib numpy
```

---

## Reference

- Mnih et al. (2013). *Playing Atari with Deep Reinforcement Learning*
- Van Hasselt et al. (2016). *Deep Reinforcement Learning with Double Q-learning*
- Wang et al. (2016). *Dueling Network Architectures for Deep Reinforcement Learning*
- Schaul et al. (2016). *Prioritized Experience Replay*
- DeepReinforcementLearning/DeepReinforcementLearningInAction (GitHub)
