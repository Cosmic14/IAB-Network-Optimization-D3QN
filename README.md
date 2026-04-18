# mmWave IAB Small-Cell Placement Optimisation via Deep Reinforcement Learning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.26%2B-013243?logo=numpy)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Abstract

This project presents a simulation framework for optimising the deployment of
millimetre-wave (mmWave) small-cell access points in an Integrated Access and
Backhaul (IAB) network operating at 60 GHz.  A **Duelling Double Deep
Q-Network (D3QN)** agent is trained to minimise capital expenditure (CAPEX)
— the number of access points deployed — while satisfying per-user throughput
constraints imposed by 60 GHz mmWave channel conditions.

The propagation environment is grounded in the LoS/NLoS pathloss equations of
**Zhang et al. (2024)**, augmented with ITU-R rain attenuation (12.6 dB/km)
and oxygen absorption (16.0 dB/km) specific to the 60 GHz band.  The agent
learns a placement policy through interaction with a custom OpenAI
Gym-compatible environment, balancing coverage quality against infrastructure
cost across stochastic urban micro-cell topologies.

---

## Tech Stack

| Component | Technology |
|---|---|
| Agent architecture | D3QN (Duelling Double DQN) |
| Deep learning framework | PyTorch 2.x |
| Numerical computation | NumPy 1.26+ |
| Channel modelling | 3GPP TR 38.901 UMi / Zhang et al. (2024) |
| Environment interface | OpenAI Gym (Gymnasium) |
| Language | Python 3.10+ |

---

## Repository Structure

```
IAB_DRL_Project/
├── channel_model.py          # mmWave LoS/NLoS pathloss & SNR computation
├── test_channel_model.py     # Unit tests for channel model
├── requirements.txt          # Python dependencies
└── README.md
```

---

## Installation

> **Prerequisites:** Python 3.10+, pip, and a virtual environment manager.

```bash
# 1. Clone the repository
git clone <repository-url>
cd IAB_DRL_Project

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

```bash
# Run channel model unit tests
python3 -m unittest test_channel_model.py -v

# Train the D3QN agent  [placeholder — to be added]
# python3 train.py

# Evaluate a saved policy  [placeholder — to be added]
# python3 evaluate.py --checkpoint checkpoints/best.pt
```

---

## Results

> _Training curves, coverage maps, and CAPEX comparison tables will be
> populated upon completion of agent training._

| Metric | Value |
|---|---|
| Mean CAPEX reduction vs. baseline | TBD |
| Mean per-user throughput | TBD |
| Coverage probability (≥ −85 dBm) | TBD |
| Training episodes to convergence | TBD |

---

## References

- Zhang et al. (2024). *mmWave Small-Cell Propagation Characterisation for
  IAB Networks at 60 GHz.* [DOI placeholder]
- 3GPP TR 38.901 V17.0.0 — *Study on channel model for frequencies from
  0.5 to 100 GHz.*
- ITU-R P.838-3 — *Specific attenuation model for rain for use in prediction
  methods.*
- ITU-R P.676-12 — *Attenuation by atmospheric gases and related effects.*
- Van Hasselt, H., Guez, A., & Silver, D. (2016). *Deep Reinforcement
  Learning with Double Q-learning.* AAAI-16.
- Wang, Z. et al. (2016). *Dueling Network Architectures for Deep
  Reinforcement Learning.* ICML-16.

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
