# mmWave IAB Small-Cell Placement Optimisation via Deep Reinforcement Learning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.26%2B-013243?logo=numpy)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Abstract

This project presents a simulation framework for optimising the sequential deployment of
millimetre-wave (mmWave) relay nodes in an **Integrated Access and Backhaul (IAB)**
network operating at **60 GHz**. A **Duelling Double Deep Q-Network (D3QN)** agent
learns a relay-placement policy that maximises user coverage while minimising capital
expenditure (CAPEX) across stochastic urban micro-cell topologies.

The propagation environment implements the **3GPP TR 38.901 UMi Street Canyon** LoS/NLoS
pathloss model, augmented with ITU-R rain attenuation (12.6 dB/km) and oxygen absorption
(16.0 dB/km) specific to the 60 GHz band. A shaped reward function prioritises per-user
coverage over raw throughput, and results are benchmarked against a static greedy heuristic
evaluated on the same fair metric (Shannon capacity ≥ demand).

---

## Tech Stack

| Component | Technology |
|---|---|
| Agent architecture | D3QN — Duelling + Double DQN |
| Deep learning | PyTorch 2.x (MPS / CUDA / CPU auto-select) |
| Numerical computation | NumPy 1.26+ |
| Channel model | 3GPP TR 38.901 UMi / Zhang et al. (2024) |
| Visualisation | Matplotlib |
| Language | Python 3.10+ |

---

## Repository Structure

```
IAB_DRL_Project/
├── environment.py        # IABEnv — Gym-style RL environment (step, reset, reward)
├── agent.py              # D3QNAgent — action selection, training step, soft update
├── network.py            # DuelingQNetwork, ReplayBuffer, device utils
├── channel_model.py      # 3GPP TR 38.901 LoS/NLoS pathloss, SNR, Shannon capacity
├── entities.py           # User, IABNode, CityGrid domain objects
│
├── train.py              # Training loop with CLI args, CSV logging, checkpointing
├── evaluate_agent.py     # Load checkpoint → run 1 episode (ε=0) → print metrics
├── baseline_greedy.py    # Static greedy heuristic baseline (fair Shannon metric)
├── plot_training.py      # 50-ep moving avg reward + epsilon decay learning curve
├── visualize.py          # Standalone deployment map renderer
│
├── models/
│   └── d3qn_checkpoint.pth   # Latest saved policy weights
├── training_log.csv          # Per-episode reward, sum-rate, nodes, epsilon, loss
│
├── learning_curve.png        # Most recent training learning curve
├── d3qn_success.png          # D3QN final deployment map
├── greedy_baseline.png       # Greedy baseline deployment map
│
├── test_agent.py             # Unit tests — D3QNAgent
├── test_channel_model.py     # Unit tests — ChannelModel
├── test_entities.py          # Unit tests — User, IABNode, CityGrid
├── test_environment.py       # Unit tests — IABEnv
├── requirements.txt
└── README.md
```

---

## Installation

> **Prerequisites:** Python 3.10+, pip, virtual environment manager.

```bash
# 1. Clone
git clone https://github.com/Cosmic14/IAB-Network-Optimization-D3QN.git
cd IAB-Network-Optimization-D3QN

# 2. Virtual environment
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# 3. Dependencies
pip install -r requirements.txt
```

---

## Usage

### Train

```bash
# Default run (10×10 action grid, 50 users, 2000 episodes)
python3 train.py --num_users 50

# Full custom run used for final results
python3 train.py --num_users 50 --episodes 8000 --n_bins 10 \
                 --epsilon_decay 0.9995 --hidden_dim 256

# Key CLI arguments
#   --num_users      UEs per episode          (default: 10)
#   --n_bins         Action grid bins/axis    (default: 10 → 100 actions)
#   --episodes       Training episodes        (default: 2000)
#   --epsilon_start  Initial exploration      (default: 1.0)
#   --epsilon_end    Minimum exploration      (default: 0.01)
#   --epsilon_decay  Per-step decay factor    (default: 0.997)
#   --hidden_dim     Network hidden width     (default: 128)
#   --lr             Adam learning rate       (default: 1e-3)
```

Checkpoints are saved to `models/d3qn_checkpoint.pth` and a per-episode log is written to `training_log.csv`.

### Evaluate the trained agent

```bash
python3 evaluate_agent.py
```

Loads `models/d3qn_checkpoint.pth`, auto-infers the architecture from the weight shapes (no manual config needed), runs one episode with **ε = 0** (pure exploitation), and prints the evaluation table. Saves the deployment map as `d3qn_success.png`.

### Run the greedy baseline

```bash
python3 baseline_greedy.py
```

Implements a static greedy heuristic (no DRL) on the same environment and seed. Coverage is reported using the **same Shannon capacity ≥ demand criterion** as the agent evaluation, making the comparison fair. Saves the deployment map as `greedy_baseline.png`.

### Plot the learning curve

```bash
python3 plot_training.py
```

Reads `training_log.csv` and saves `learning_curve.png` — a dual-axis plot showing the 50-episode moving average reward (primary Y) and epsilon decay (secondary Y) against episode number.

### Run unit tests

```bash
python3 -m unittest discover -v
```

---

## Environment Design

### Simulation area
A **1000 m × 1000 m** grid with 10 m resolution, consistent with Zhang et al. (2023) mmWave dense-urban deployment scenarios.

### Nodes
- **Donors** — 5–20 per episode, placed randomly at discrete 10 m candidate sites. Fibre-connected; unlimited upstream capacity (10 000 Mbps).
- **Relays** — placed sequentially by the agent, one per step. Wireless backhaul to the nearest donor; subject to the backhaul capacity constraint `flow_in ≥ 1.2 × (access_demand + flow_out)`.

### State vector (flattened, float32)
```
user_positions   (num_users × 2)
node_positions   (MAX_NODES × 2, zero-padded)
donor_position   (2)
num_nodes        (1)
sum_rate_mbps    (1)
coverage_rate    (1)
Total: 2·num_users + 2·MAX_NODES + 5
```

### Action space
An **n_bins × n_bins** discrete grid over the 1000 m area. Each action maps to a cell centre coordinate where the next relay is placed. With `n_bins=10`: **100 actions**.

### Reward function (shaped)
```
R = 0.1 × sum_rate_Mbps                   # throughput — secondary signal
  − 100  × num_backhaul_violations
  − 50   × num_relay_nodes                # CAPEX penalty
  + 500  × newly_connected_users          # coverage — primary signal
  − 100  × unconnected_users  [if done]   # terminal coverage penalty
```

The `+500` per newly-covered user was introduced to fix reward hacking — previously the agent maximised raw throughput without improving user coverage.

---

## Agent Architecture (D3QN)

```
Input (state_dim)
    │
    ├── FC(state_dim → hidden_dim) + ReLU
    └── FC(hidden_dim → hidden_dim) + ReLU    ← shared trunk
                │
      ┌─────────┴──────────┐
      │                    │
 Value stream        Advantage stream
 FC(hidden → 1)      FC(hidden → action_dim)
      │                    │
      └────── Q(s,a) = V(s) + [A(s,a) − mean A(s,·)] ──────┘
```

- **Double DQN** — policy net selects next action; target net evaluates it (decouples maximisation bias).
- **Polyak target update** — `τ = 0.005` soft update every step.
- **Huber loss** with gradient clipping (`max_norm = 10`).
- **Experience replay** — uniform sampling from a 10 000-transition buffer.

---

## Results

All metrics use **Shannon capacity ≥ 100 Mbps per user** as the coverage criterion — the same threshold applied to both the greedy baseline and the D3QN agent.

### Greedy baseline vs D3QN (seed=42, 50 users)

| Method | Coverage (Shannon ≥ 100 Mbps) | Sum-Rate | Relays Deployed |
|---|---|---|---|
| Greedy heuristic (no relays) | 6 / 50 | 1 717 Mbps | 0 |
| D3QN — 7×7, 2 000 ep, no shaping | 12 / 50 | 3 618 Mbps | 10 |
| D3QN — 7×7, 3 000 ep, shaped (0.5×) | 14 / 50 | 4 707 Mbps | 10 |
| D3QN — 15×15, 3 000 ep, shaped (0.5×) | 12 / 50 | 3 937 Mbps | 10 |
| **D3QN — 10×10, 8 000 ep, shaped (0.1×)** | **13 / 50** | **4 967 Mbps** | **10** |

> **Note on the greedy's apparent "43/50":** An earlier version of the baseline counted users within 200 m of a donor as "connected." At 60 GHz, NLoS attenuation at 100 m yields ≈ 0.6 Mbps — far below the 100 Mbps demand — making that metric meaningless. The table above uses the corrected Shannon-capacity criterion throughout.

### Why full coverage is physically constrained

At 60 GHz, NLoS pathloss is severe:

| Distance | LoS capacity | NLoS capacity |
|---|---|---|
| 30 m | 351 Mbps | 56 Mbps |
| 50 m | 204 Mbps | 10 Mbps |
| 100 m | 61 Mbps | 0.6 Mbps |
| 200 m | 9 Mbps | ≈ 0 Mbps |

A user in NLoS at 50 m receives only 10 Mbps regardless of relay placement. Achieving 100% coverage at 100 Mbps requires reducing the demand threshold, increasing TX power, or reducing the grid size — all of which are configurable parameters.

---

## References

- Zhang et al. (2024). *mmWave Small-Cell Propagation Characterisation for IAB Networks at 60 GHz.*
- 3GPP TR 38.901 V17.0.0 — *Study on channel model for frequencies from 0.5 to 100 GHz.*
- ITU-R P.838-3 — *Specific attenuation model for rain for use in prediction methods.*
- ITU-R P.676-12 — *Attenuation by atmospheric gases and related effects.*
- Van Hasselt, H., Guez, A., & Silver, D. (2016). *Deep Reinforcement Learning with Double Q-learning.* AAAI-16.
- Wang, Z. et al. (2016). *Dueling Network Architectures for Deep Reinforcement Learning.* ICML-16.
- Mnih, V. et al. (2015). *Human-level control through deep reinforcement learning.* Nature.

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
