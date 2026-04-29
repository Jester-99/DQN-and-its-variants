# HW3-1: Understanding Report — Naive DQN

## 1. What is DQN?

**Deep Q-Network (DQN)** combines Q-learning with a deep neural network to
approximate the action-value function Q(s, a). The key Bellman update target is:

```
y = r  +  γ · max_{a'} Q(s', a')      if s' is not terminal
y = r                                  otherwise
```

The network is trained to minimize the mean-squared TD-error:
```
L(θ) = E[ (y − Q(s, a; θ))² ]
```

---

## 2. Experience Replay Buffer

### What it is
A fixed-size circular queue that stores past transitions:
```
(state, action, reward, next_state, done)
```

### Why it is necessary
Without a replay buffer the agent trains on consecutive transitions, which are
**highly correlated** (consecutive game frames look almost the same). Training
on correlated data makes gradient descent unstable and causes the Q-network to
"forget" earlier experiences.

By sampling a **random mini-batch** from the buffer we:

| Problem without replay | Solution with replay |
|---|---|
| Correlated data → biased gradients | Mini-batch breaks correlation |
| Each experience used once | Each transition replayed many times |
| Non-stationary distribution | Buffer stabilises the data distribution |

### Buffer capacity
A capacity of 10,000 is typical for small environments. The agent starts
learning only when the buffer has at least `BATCH_SIZE` (64) transitions.

---

## 3. ε-Greedy Exploration

The agent picks:
- A **random action** with probability ε  (exploration)
- The **greedy action** (argmax Q) otherwise (exploitation)

ε decays from 1.0 → 0.01 over training, shifting from pure exploration to
mostly exploitation as the Q-function matures.

---

## 4. GridWorld (Static Mode)

```
G  P  -  +        G=Goal, P=Pit, W=Wall (row 1), +=Player
-  W  -  -
```

Player starts at (0,3). Goal at (0,0), Pit at (0,1), Wall at (1,1).

Rewards:
- Reach Goal → **+10** (terminal)
- Fall in Pit → **−10** (terminal)
- Any other step → **−1** (step penalty encourages shortest path)

---

## 5. Observations from Training

- In static mode convergence is fast (<200 episodes) because the state space
  is identical every episode.
- The replay buffer helps even in this small environment by providing
  decorrelated training samples.
- ε-decay is the primary driver of the reward increase curve.

---

## 6. Conversation with ChatGPT (summary)

**Q:** Why does DQN use a separate target network?

**A (ChatGPT):** The target network is a periodically-updated copy of the
online Q-network. Without it, the TD target y = r + γ·max Q(s',a') would
change every gradient step because the same network computes both the
prediction and the target. This "moving target" problem causes instability.
By freezing the target network for a number of steps, the targets remain
stable, making gradient descent converge more reliably.

**Q:** What is the difference between on-policy and off-policy learning?

**A (ChatGPT):** DQN is **off-policy**: it stores transitions from a
potentially old policy (stored in the replay buffer) and learns a different
(greedy) policy. This is why the replay buffer works — we can reuse old data.
SARSA, by contrast, is **on-policy**: it can only learn from data generated
by its current policy.
