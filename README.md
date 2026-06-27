# Snake Reinforcement Learning

A from-scratch **Deep Q-Learning (DQN)** system that teaches an AI agent to play the classic game of Snake — no hardcoded rules, no scripted paths. The agent perceives the board through an 11-dimensional state vector, chooses from three relative moves, and learns purely from rewards (+10 for food, −10 for dying) by optimizing the weights of a PyTorch neural network. The project pairs a custom Pygame Snake environment and a Deep Q-Network trainer with an interactive **Streamlit web app** that streams training live in the browser: watch the snake play, see the score/mean-score curve climb in real time, tune the playback speed, and follow the exploration rate decay as the network shifts from random flailing to deliberate, food-seeking play.

## Check out the website [here](https://snake-reinforcement-learning.streamlit.app/)!

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.58%2B-FF4B4B.svg)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12-EE4C2C.svg)](https://pytorch.org/)
[![RL](https://img.shields.io/badge/RL-Deep%20Q--Learning-success.svg)](https://en.wikipedia.org/wiki/Q-learning)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Setup](#setup)
3. [Repository Layout](#repository-layout)
4. [How Reinforcement Learning Works](#how-reinforcement-learning-works)
5. [The Game Environment](#the-game-environment)
6. [State Representation](#state-representation)
7. [Action Space](#action-space)
8. [Reward Function](#reward-function)
9. [The Neural Network](#the-neural-network)
10. [The Training Step (The Math in Code)](#the-training-step-the-math-in-code)
11. [The Agent](#the-agent)
12. [Two Ways to Run](#two-ways-to-run)
13. [Running the Streamlit App](#running-the-streamlit-app)
14. [Deploying to Streamlit Cloud (Engineering Notes)](#deploying-to-streamlit-cloud-engineering-notes)
15. [Hyperparameters](#hyperparameters)
16. [Limitations & Possible Improvements](#limitations--possible-improvements)
17. [Appendix](#appendix)

---

## Project Overview

### What It Does

Snake Reinforcement Learning is an end-to-end deep reinforcement learning system that:

- **Simulates Snake** in a custom Pygame environment purpose-built for an AI to play (no keyboard input, deterministic step API).
- **Perceives the board** as an 11-dimensional binary state vector (immediate danger, current direction, relative food location).
- **Acts** using three *relative* moves — go straight, turn right, turn left.
- **Learns** with a Deep Q-Network: a small PyTorch multilayer perceptron that estimates the long-term value of each action and is trained via the Bellman equation.
- **Balances exploration and exploitation** with a decaying epsilon-greedy policy (random early, deliberate later).
- **Stabilizes learning** using experience replay — short-term training after every single step plus long-term batch training over 1,000 past experiences at the end of each game.
- **Visualizes everything live** in a Streamlit dashboard: the snake playing in real time, a score/mean-score training curve, and live metrics (games played, high score, current score, exploration rate).

### Architecture Overview

```
                ┌──────────────────────────────────────────────┐
                │                  AGENT (agent.py)             │
                │   get_state() → get_action() → remember()     │
                │      │              │             │           │
                │      │     ┌────────┴────────┐    │           │
                │      │     │  Linear_QNet     │    │           │
                │      │     │  11 → 256 → 3    │    │           │
                │      │     │   (model.py)     │    │           │
                │      │     └────────┬────────┘    │           │
                │      │              │             │           │
                │      │       QTrainer.train_step  │           │
                │      │     (Bellman + MSE + Adam) │           │
                └──────┼──────────────┼─────────────┼───────────┘
                 state │       action │      reward,│ next_state
                       │              v             │
                ┌──────┴──────────────────────────────────────┐
                │            ENVIRONMENT (game.py)             │
                │   SnakeGameAI.play_step(action)              │
                │   → (reward, game_over, score)               │
                └──────────────────────────────────────────────┘

   Run it two independent ways:
   ┌─────────────────────────┐        ┌──────────────────────────────┐
   │  python agent.py        │        │  streamlit run app.py        │
   │  (Pygame window +       │        │  (browser dashboard, live    │
   │   live matplotlib plot) │        │   board + chart + controls)  │
   └─────────────────────────┘        └──────────────────────────────┘
```

### Problems Solved

1. **Learning control from scratch** — the agent starts knowing nothing and discovers a food-seeking, collision-avoiding policy purely from trial, error, and reward.
2. **Generalizable state design** — a compact, relative 11-dimensional state lets a tiny network learn quickly without memorizing absolute board positions.
3. **Training stability** — experience replay and a decaying exploration schedule prevent the network from chasing noisy, correlated updates.
4. **Accessible visualization** — the Streamlit app turns an abstract training loop into something anyone can watch and interact with in a browser, with no install.

---

## Setup

### Prerequisites

- **Python 3.10+** (developed/tested on 3.13)
- **pip** (package manager)
- **Git** (for cloning)

### macOS / Linux

```bash
# Clone repository
git clone https://github.com/TejasNaik24/Snake-Reinforcement-Learning.git
cd Snake-Reinforcement-Learning

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Option A: run the standalone Pygame trainer
python agent.py

# Option B: run the Streamlit web app
streamlit run app.py
```

### Windows (PowerShell)

```powershell
# Clone repository
git clone https://github.com/TejasNaik24/Snake-Reinforcement-Learning.git
cd Snake-Reinforcement-Learning

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Option A: run the standalone Pygame trainer
python agent.py

# Option B: run the Streamlit web app
streamlit run app.py
```

### Expected Output

- **`python agent.py`** opens a Pygame window showing the snake training, prints `Game N Score X Record: Y` to the console each game, and shows a live matplotlib plot of score and mean score.
- **`streamlit run app.py`** opens your browser to `http://localhost:8501` with a landing page → a **Train Model** button → a live training dashboard.

---

## Repository Layout

```
Snake-Reinforcement-Learning/
├── agent.py            # The RL agent: state, epsilon-greedy actions, memory, training loop
├── model.py            # The neural network (Linear_QNet) + the DQN trainer (QTrainer)
├── game.py             # The Pygame Snake environment built for an AI to play
├── helper.py           # Live matplotlib plotting for the standalone trainer
├── app.py              # The Streamlit web app (live training dashboard)
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── model/
    └── model.pth       # Saved network weights (generated by training; gitignored)
```

> **Note:** `model/model.pth` is **not** committed to the repo — `*.pth` is gitignored. It is created automatically the first time the agent beats its own high score (`model.save()`).

### Core Files

#### `agent.py`

The reinforcement learning **agent** and the standalone training loop. Key pieces:

- `class Agent` — holds the network, the trainer, the replay memory (`deque(maxlen=100_000)`), and the bookkeeping (`n_games`, `epsilon`, `gamma`).
- `get_state(game)` — converts the live game into the **11-dimensional state vector**.
- `get_action(state)` — the **epsilon-greedy policy**: a random move while exploring, otherwise the network's best (argmax) move.
- `remember(...)` — appends `(state, action, reward, next_state, done)` to replay memory.
- `train_short_memory(...)` — trains on the single most recent transition (every step).
- `train_long_memory()` — samples a batch of 1,000 transitions and trains on them (every game over) = **experience replay**.
- `train()` + `if __name__ == '__main__'` — the standalone Pygame training entry point.

#### `model.py`

The **neural network** and the **DQN training math**:

- `class Linear_QNet(nn.Module)` — a two-layer MLP: `Linear(11, 256) → ReLU → Linear(256, 3)`. Includes `save()` which writes `./model/model.pth`.
- `class QTrainer` — wraps the Bellman update. Uses the **Adam** optimizer (`lr=0.001`) and **mean squared error** loss. Its `train_step()` computes Q-targets and backpropagates.

#### `game.py`

The **environment**: a Snake game adapted for AI control (`class SnakeGameAI`). Instead of reading the keyboard, it exposes `play_step(action)` which advances one frame and returns `(reward, game_over, score)`. Also handles food placement, collisions, the move-decoding logic, and rendering to a Pygame surface.

#### `helper.py`

A single `plot(scores, mean_scores)` function that draws the live score / mean-score curve during standalone training (uses `IPython.display`, ideal in notebooks and Pygame runs).

#### `app.py`

The **Streamlit web app** — a self-contained live training dashboard (landing page, train/pause/resume/reset controls, live board, live chart, speed slider, win dialog). It reuses `Agent` and `SnakeGameAI` unchanged and renders the board as lightweight HTML. See [Running the Streamlit App](#running-the-streamlit-app) and [Deploying to Streamlit Cloud](#deploying-to-streamlit-cloud-engineering-notes).

---

## How Reinforcement Learning Works

This section is the conceptual + mathematical core. Every formula below maps directly onto the code in `agent.py` and `model.py`.

### 1. The Fundamentals

**Reinforcement Learning (RL)** is learning by interacting with an environment to maximize cumulative reward. The vocabulary:

| Term | Meaning | In this project |
|------|---------|-----------------|
| **Agent** | The decision maker | `Agent` in `agent.py` |
| **Environment** | The world the agent acts in | `SnakeGameAI` in `game.py` |
| **State (s)** | A snapshot of the situation | 11-dim vector from `get_state()` |
| **Action (a)** | A choice the agent makes | `[straight, right, left]` |
| **Reward (r)** | Feedback signal | +10 food, −10 death, 0 otherwise |
| **Policy (π)** | Strategy mapping states → actions | epsilon-greedy over the network |
| **Episode** | One full game (reset → game over) | one Snake game |

### 2. The Agent–Environment Loop

Reinforcement learning is a loop. At each step the agent observes the state, picks an action, and the environment responds with a reward and a new state:

```
   ┌────────────► state sₜ ───────────┐
   │                                  ▼
ENVIRONMENT                        AGENT
(game.py)                         (agent.py)
   ▲                                  │
   │                          action aₜ
   └──── reward rₜ, state sₜ₊₁ ◄──────┘
```

The agent's goal is **not** to maximize the immediate reward, but the *total future reward* — which is why it needs to value actions that pay off later (like positioning itself to reach food without trapping itself).

### 3. Q-Learning and the Q-Function

The central idea of **Q-Learning** is the **Q-function** `Q(s, a)`: the expected total future reward of taking action `a` in state `s` and then acting optimally thereafter. If you knew `Q(s, a)` perfectly, the optimal policy would be trivial — in any state, pick the action with the highest Q-value:

```
π*(s) = argmax_a  Q(s, a)
```

In classic tabular Q-Learning you'd store `Q(s, a)` in a giant table. That works for small problems, but Snake's state space (every configuration of snake + food) is far too large to enumerate — which is where deep learning comes in.

### 4. The Bellman Equation (the key math)

How do we learn `Q(s, a)`? With the **Bellman equation**, which expresses a Q-value recursively — the value of acting now equals the immediate reward plus the (discounted) best value available from the next state:

```
Q(s, a)  =  r  +  γ · max_a'  Q(s', a')
```

- `r` — the immediate reward.
- `s'` — the next state.
- `max_a' Q(s', a')` — the best achievable value from the next state.
- **γ (gamma)** — the **discount factor**, between 0 and 1, controlling how much future reward matters. This project uses **γ = 0.9** (`agent.py`), so the agent strongly values the future but slightly prefers sooner rewards. γ = 0 would make it greedy/short-sighted; γ → 1 makes it far-sighted.

If the step ends the game (`done`), there is no future, so the target collapses to just `Q(s, a) = r`.

### 5. Deep Q-Learning (DQN)

Instead of a table, we approximate the Q-function with a **neural network** `Q(s, a; θ)` whose weights are `θ`. The network takes the 11-dimensional state and outputs **three numbers** — one estimated Q-value per action:

```
state (11)  ─►  [ neural network θ ]  ─►  [Q_straight, Q_right, Q_left]
```

Training nudges the weights `θ` so the network's predictions match the Bellman targets.

### 6. Loss and Gradient Descent

We turn the Bellman equation into a supervised learning problem. For the action the agent actually took, the **target** is the Bellman value; the **prediction** is the network's current estimate. We minimize the **mean squared error** between them:

```
L(θ)  =  ( Q_target  −  Q_pred )²
      =  ( [r + γ · max_a' Q(s', a')]  −  Q(s, a; θ) )²
```

This is exactly what `QTrainer` does: `criterion = nn.MSELoss()`, optimized with **Adam** at **learning rate 0.001**. Backpropagation computes the gradient of `L` with respect to `θ`, and Adam takes a step to reduce it. See [The Training Step](#the-training-step-the-math-in-code) for the line-by-line mapping.

### 7. Exploration vs. Exploitation (epsilon-greedy)

If the agent always picked its current best action, it would never discover better ones. If it always acted randomly, it would never *use* what it learned. The classic resolution is **epsilon-greedy**: act randomly with probability ε, otherwise act greedily.

This project uses a **decaying** schedule (`agent.py`):

```python
self.epsilon = 80 - self.n_games
if random.randint(0, 200) < self.epsilon:
    # explore: random move
else:
    # exploit: argmax of the network's Q-values
```

- At **game 0**: `epsilon = 80`, so P(random) ≈ 80 / 201 ≈ **40%** — lots of exploration.
- The exploration probability **decreases linearly** with each game played.
- By **game 80**: `epsilon ≤ 0`, so the agent stops exploring and acts purely on what it has learned.

This naturally transitions the agent from "try everything" to "exploit your knowledge" as it gains experience.

### 8. Experience Replay (two memories)

Training a network on a stream of *consecutive* game frames is unstable — back-to-back frames are highly correlated. **Experience replay** fixes this by storing transitions and training on them in mixes. This project uses two complementary levels:

- **Short-term memory** (`train_short_memory`) — after **every single step**, train once on just that transition so the agent reacts immediately.
- **Long-term memory** (`train_long_memory`) — at **every game over**, randomly sample a batch of **1,000** transitions from a replay buffer of up to **100,000** and train on them together. Random sampling breaks the correlation between consecutive frames and reuses past experience efficiently.

```python
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
self.memory = deque(maxlen=MAX_MEMORY)   # old transitions drop off automatically
```

---

## The Game Environment

The environment lives in `game.py` as `class SnakeGameAI`. It is a standard Snake game with one crucial change: it is driven by an **action array** rather than the keyboard, so an AI can play it.

| Property | Value |
|----------|-------|
| Board size | 640 × 480 pixels |
| `BLOCK_SIZE` | 20 px → a **32 × 24** grid |
| Starting snake | length 3, moving RIGHT, centered |
| Food | placed at a random empty cell |
| `SPEED` | 40 (frame cap, used in the standalone Pygame run only) |

### The `play_step` contract

```python
reward, game_over, score = game.play_step(action)
```

Each call advances exactly one frame: it moves the snake per `action`, checks for game over, places food or moves on, redraws, and returns the three values. This tight, single-step API is what makes the environment trainable.

### Game over conditions

The game ends (`game_over = True`, `reward = -10`) when either:

1. **Collision** — the head hits a wall or the snake's own body (`is_collision()`), **or**
2. **Timeout** — `frame_iteration > 100 * len(snake)`. This prevents the agent from earning a "safe" infinite score by looping forever without eating; it must keep making progress.

---

## State Representation

The agent never sees raw pixels. Instead, `get_state(game)` in `agent.py` compresses the board into an **11-dimensional binary vector** — the network's entire view of the world:

```
[ danger_straight, danger_right, danger_left,        # 3 → immediate collision danger
  dir_left, dir_right, dir_up, dir_down,             # 4 → current direction (one-hot)
  food_left, food_right, food_up, food_down ]        # 4 → where food is, relative to head
```

- **Danger (3)** — is there a collision one cell straight ahead, to the right, or to the left of the current heading? Computed by probing the cell in each relative direction with `is_collision()`.
- **Direction (4)** — a one-hot flag for the snake's current heading.
- **Food (4)** — boolean comparisons of the food's position vs. the head's position (`food.x < head.x`, etc.).

### Worked example

Suppose the snake is moving **RIGHT**, there's a wall immediately to its right, and the food is up-and-to-the-left:

```
danger_straight = 0   # clear ahead
danger_right    = 1   # wall if it turns right
danger_left     = 0   # clear if it turns left
dir_left=0, dir_right=1, dir_up=0, dir_down=0
food_left=1, food_right=0, food_up=1, food_down=0
→ state = [0,1,0, 0,1,0,0, 1,0,1,0]
```

### Why a *relative* state?

Because danger and direction are encoded **relative to the snake's heading** (not as absolute "north/south/east/west"), the same learned behavior transfers no matter which way the snake is facing. This keeps the state tiny and lets an 11-input network learn a general policy fast.

---

## Action Space

The agent chooses one of **three relative moves**, encoded as a one-hot array:

| Action | Meaning |
|--------|---------|
| `[1, 0, 0]` | Go **straight** (keep current direction) |
| `[0, 1, 0]` | Turn **right** |
| `[0, 0, 1]` | Turn **left** |

Internally, `_move()` in `game.py` decodes these against a clockwise direction ring `[RIGHT, DOWN, LEFT, UP]`:

- Straight → keep the same index.
- Right turn → `(index + 1) % 4`.
- Left turn → `(index - 1) % 4`.

### Why relative, not absolute?

There is deliberately **no "reverse" action**, so the snake can never instantly turn 180° into its own neck. Relative actions also pair naturally with the relative state — together they make the control problem symmetric and far easier to learn than four absolute directions.

---

## Reward Function

Rewards are the only feedback the agent receives. The scheme (in `game.py`) is intentionally simple:

| Event | Reward |
|-------|--------|
| Eats food | **+10** |
| Game over (collision or timeout) | **−10** |
| Any other step | **0** |

The sparse, high-contrast signal (+10 / −10) gives a clear gradient: seek food, avoid death. The timeout penalty is what discourages aimless survival loops and pushes the agent to actively pursue food.

---

## The Neural Network

Defined in `model.py` as `Linear_QNet` — a compact two-layer feedforward network (multilayer perceptron):

```
 input          hidden            output
(state, 11) ─► Linear(11,256) ─► ReLU ─► Linear(256,3) ─► (Q-values, 3)
```

```python
class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)   # 11 → 256
        self.linear2 = nn.Linear(hidden_size, output_size)  # 256 → 3

    def forward(self, x):
        x = F.relu(self.linear1(x))   # non-linearity
        x = self.linear2(x)           # raw Q-values (no activation)
        return x
```

- **Input layer:** 11 neurons (the state vector).
- **Hidden layer:** 256 neurons with **ReLU** activation.
- **Output layer:** 3 neurons — one **Q-value per action**. No activation on the output, because Q-values are unbounded real numbers, not probabilities.

The model is instantiated as `Linear_QNet(11, 256, 3)` in `agent.py`. Calling `model.save()` writes the weights (`state_dict`) to `./model/model.pth`.

---

## The Training Step (The Math in Code)

`QTrainer.train_step()` in `model.py` is where the Bellman equation becomes gradient descent. Here is the core, annotated:

```python
# 1. What the network currently predicts for this state
pred = self.model(state)                      # Q(s, ·)

# 2. Build the target = a copy of pred, with the taken action's value replaced
target = pred.clone()
for idx in range(len(done)):
    Q_new = reward[idx]                       # if the game ended: target = r
    if not done[idx]:
        Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
        #        └─────────── Bellman: r + γ · max_a' Q(s', a') ───────────┘
    target[idx][torch.argmax(action[idx]).item()] = Q_new

# 3. Push predictions toward the target and update the weights
self.optimizer.zero_grad()
loss = self.criterion(target, pred)           # MSE( target, prediction )
loss.backward()                               # backpropagation
self.optimizer.step()                         # Adam update
```

Key details:

- **Only the taken action's Q-value is changed** in the target. The other two outputs are copied straight from `pred`, so their error is zero and they aren't trained on this step — we only have evidence about the action we actually tried.
- **Terminal handling:** if `done`, there's no next state, so `Q_new = reward` (the future term drops out).
- **Single vs. batch:** the method reshapes a single transition into a batch of one (`torch.unsqueeze(..., 0)`), so the *same code* powers both `train_short_memory` (one sample) and `train_long_memory` (1,000 samples).
- **Optimizer/loss:** `Adam(lr=0.001)` and `nn.MSELoss()`.

---

## The Agent

`agent.py` ties everything together. The per-step training cycle (in `train()`, and mirrored inside the Streamlit loop) is:

1. **Observe** — `state_old = agent.get_state(game)`.
2. **Decide** — `final_move = agent.get_action(state_old)` (epsilon-greedy).
3. **Act** — `reward, done, score = game.play_step(final_move)`, then `state_new = agent.get_state(game)`.
4. **Learn (short)** — `agent.train_short_memory(state_old, final_move, reward, state_new, done)`.
5. **Remember** — `agent.remember(state_old, final_move, reward, state_new, done)`.
6. **On game over** — `game.reset()`, increment `n_games`, run `agent.train_long_memory()` (batch replay), save the model if it's a new record, and log the score.

This repeats indefinitely; over hundreds of games the score curve trends upward as the network's Q-estimates sharpen and epsilon decays toward pure exploitation.

---

## Two Ways to Run

The project intentionally supports **two completely independent** entry points. The Streamlit app does **not** modify or depend on the Pygame training path — clone the repo and use either.

### A. Standalone Pygame Trainer

```bash
python agent.py
```

Opens a native Pygame window, trains continuously, prints per-game scores to the console, and shows a live matplotlib plot via `helper.plot()`. This is the original, classic training loop.

### B. Streamlit Web App

```bash
streamlit run app.py
```

Runs the browser dashboard described below. It renders the game headlessly (no desktop window) and streams the board, chart, and metrics to your browser.

---

## Running the Streamlit App

### UI Flow

1. **Landing page** — a project overview card and a single **Train Model** button.
2. **Training dashboard** — appears after you click Train Model:
   - **Game Screen** — the snake playing live, rendered as lightweight HTML.
   - **Training Metrics** — a live score / mean-score curve (the network's progress).
   - **Metric bar** — Games Played, High Score, Current Score, and **Epsilon (Randomness %)**.
   - **Controls** — Pause / Resume / Reset Training / Back to Home.
   - **Speed slider** — "Snake Speed (Frames per Second)", 1–30, to slow the snake down to watch it think or speed it up to train faster.
3. **Win dialog** — if the snake ever fills the board (a perfect game), a celebratory modal shows final stats with **Close** and **Retrain** options.

### Reading the metrics

- **Epsilon (Randomness %)** starts high and decays toward 0 — a live view of the exploration→exploitation transition described above.
- **Mean Score** rising over many games is the clearest signal the agent is actually learning (individual game scores are noisy).

Because the agent, game, and score history are kept in `st.session_state`, moving the speed slider mid-session **does not reset training** — it only changes playback speed.

---

## Deploying to Streamlit Cloud (Engineering Notes)

Getting a real-time Pygame-based RL loop to run smoothly on Streamlit Cloud required solving several non-obvious problems. These notes document what and why.

### 1. Headless Pygame

Streamlit Cloud has no display. Pygame is forced into a dummy video driver **before it is imported**:

```python
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"   # must run before pygame is imported
```

A `DummyClock` also replaces Pygame's `clock.tick()` so the internal frame cap never blocks the Streamlit loop.

### 2. `pygame-ce` instead of `pygame`

Streamlit Cloud's installer uses a very new Python where stock `pygame` has no prebuilt wheel and tries (and fails) to compile from source against missing SDL headers. The fix is **`pygame-ce`** (the actively-maintained community fork) which ships prebuilt wheels — it's a drop-in replacement that still `import pygame`s, so `game.py` needs no changes.

### 3. HTML rendering instead of image streaming

The biggest performance issue: encoding a full 640×480 **PNG every frame** and streaming it over the websocket chokes the cloud's single shared CPU core, producing ~1 fps "teleporting" motion (while looking perfectly smooth on localhost). The fix is `game_to_html()`, which draws the board as a handful of absolutely-positioned `<div>`s — a few KB of text, no image codec, rendered natively by the browser.

> Note: divs, not SVG. `st.html` sanitizes with **DOMPurify's html-only profile**, which strips `<svg>`/`<rect>` elements but keeps styled `<div>`s.

### 4. Coupled, frame-paced loop

The training loop renders exactly **one game step per frame** and paces each frame to a steady wall-clock budget (`1 / speed`), subtracting time already spent so an expensive step (like end-of-game batch training) doesn't compound. This guarantees the snake always advances one cell per frame (never teleports) and the motion stays even.

### 5. Throttled chart redraws

The matplotlib chart also encodes a PNG, so it is redrawn at most once per second instead of on every game over, smoothing out hitches on the shared core.

---

## Hyperparameters

| Hyperparameter | Value | Where | Meaning |
|----------------|-------|-------|---------|
| State size | 11 | `agent.py` | Input dimension of the network |
| Hidden size | 256 | `agent.py` | Hidden-layer width |
| Action size | 3 | `agent.py` | Straight / right / left |
| Discount factor (γ) | 0.9 | `agent.py` | Weight on future reward in the Bellman target |
| Learning rate | 0.001 | `agent.py` / `model.py` | Adam step size |
| Batch size | 1,000 | `agent.py` | Long-term replay batch |
| Max memory | 100,000 | `agent.py` | Replay buffer capacity |
| Epsilon schedule | `80 − n_games` | `agent.py` | Linear exploration decay (0 by game 80) |
| Reward (food) | +10 | `game.py` | Eating food |
| Reward (death) | −10 | `game.py` | Collision or timeout |
| Optimizer | Adam | `model.py` | Gradient-based weight updates |
| Loss | MSE | `model.py` | Bellman target vs. prediction |
| `BLOCK_SIZE` | 20 px | `game.py` | Grid cell size (32×24 board) |

---

## Limitations & Possible Improvements

### Limitations

1. **Hand-crafted state** — the 11-dim vector encodes only immediate danger, heading, and food direction. The agent is effectively "blind" beyond one cell ahead, so it can trap itself with its own long tail.
2. **Simple network** — a small MLP, not a convolutional/vision model. It cannot "see" the full board layout.
3. **Vanilla DQN** — no target network, no Double DQN, no dueling architecture, no prioritized replay. These are known stabilizers/improvers omitted for simplicity.
4. **No reward shaping** — rewards are only ±10/0; intermediate guidance (e.g., reward for moving toward food) could speed learning.
5. **Deterministic inference** — actions are taken by `argmax` (no stochastic sampling at play time).
6. **Cloud training speed** — on Streamlit Cloud's shared single core, training is intentionally paced for smooth visuals, so it learns much slower than a local multi-core run.

### Possible Improvements

1. **Richer state** — feed the full grid (and use a CNN), or add tail-distance / free-space features to avoid self-trapping.
2. **Modern DQN variants** — add a **target network** and **Double DQN** to reduce overestimation and stabilize learning.
3. **Prioritized experience replay** — sample surprising transitions more often.
4. **Reward shaping** — small positive/negative rewards for getting closer to / farther from food.
5. **Hyperparameter tuning** — sweep γ, learning rate, hidden size, and the epsilon schedule.
6. **Checkpoint loading in the app** — let the Streamlit app load a pretrained `model.pth` to demo a skilled snake instantly.

---

## Appendix

### A. A Worked Bellman Update

Suppose the snake takes the **straight** action `[1, 0, 0]`, eats food (so `reward = 10`), and the game does **not** end. Let the network's current outputs be:

```
pred                = Q(s, ·)  = [2.0, 1.5, 0.3]      # [straight, right, left]
max_a' Q(s', a')    = 5.0                              # best value from the next state
γ                   = 0.9
```

The Bellman target for the taken action (straight) is:

```
Q_new = r + γ · max_a' Q(s', a')
      = 10 + 0.9 × 5.0
      = 14.5
```

So the target vector becomes the prediction with **only the straight slot replaced**:

```
target = [14.5, 1.5, 0.3]      # right/left untouched (no evidence this step)
```

The loss is `MSE(target, pred)`, dominated by the straight slot's error `(14.5 − 2.0)² = 156.25`. Backprop + Adam then nudge the weights so the next prediction for `Q(s, straight)` moves toward 14.5. If instead the step had ended the game (`done`), the target would be just `Q_new = reward = 10` (no future term).

### B. Example Standalone Training Output

```
Game 1 Score 0 Record: 0
Game 2 Score 0 Record: 0
Game 3 Score 1 Record: 1
Game 4 Score 0 Record: 1
...
Game 80 Score 18 Record: 23
Game 120 Score 31 Record: 41
```

Early games are near-random (high epsilon); scores and the record climb as the network learns and exploration decays.

### C. Credits

This project is built on the excellent **freeCodeCamp** tutorial *"Teach AI To Play Snake — Reinforcement Learning Tutorial With PyTorch And Pygame"* (the `agent.py` / `model.py` / `game.py` / `helper.py` Deep Q-Learning implementation). The **Streamlit web app** (`app.py`), the live in-browser visualization, and the cloud-deployment engineering are original additions on top of that foundation.
