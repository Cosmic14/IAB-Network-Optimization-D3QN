"""
network.py

Duelling Double Deep Q-Network (D3QN) architecture and experience replay
buffer for the mmWave IAB small-cell placement optimisation agent.

Architecture reference:
    Wang et al. (2016) "Dueling Network Architectures for Deep Reinforcement
    Learning", ICML.  Combined with Double DQN (van Hasselt et al., 2016)
    during training to reduce Q-value over-estimation.
"""

from __future__ import annotations

import collections
import random
from typing import Deque, NamedTuple, Tuple

import numpy as np
import torch
import torch.nn as nn


# ── Device utility ────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """
    Return the best available compute device.

    Priority order: CUDA (NVIDIA GPU) → MPS (Apple Silicon) → CPU.

    Returns
    -------
    torch.device
        The selected device, ready to pass to ``.to(device)``.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Transition container ──────────────────────────────────────────────────────

class Transition(NamedTuple):
    """Single (s, a, r, s', done) experience tuple stored in the replay buffer."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


# ══════════════════════════════════════════════════════════════════════════════
#  Duelling Q-Network
# ══════════════════════════════════════════════════════════════════════════════

class DuelingQNetwork(nn.Module):
    """
    Duelling Deep Q-Network with shared feature extraction and separate
    Value / Advantage streams.

    Architecture
    ------------
    Input (input_dim)
        │
        ├── FC(input_dim → hidden_dim) + ReLU
        └── FC(hidden_dim → hidden_dim) + ReLU   ← shared trunk
                    │
          ┌─────────┴──────────┐
          │                    │
    Value stream          Advantage stream
    FC(hidden → 1)        FC(hidden → action_dim)
          │                    │
          └────────── Q ───────┘
                  Q(s,a) = V(s) + [A(s,a) − mean_a A(s,a)]

    The mean-centering of the advantage stream (rather than max) preserves
    the identifiability of V and A while remaining differentiable everywhere
    (Wang et al., 2016).

    Parameters
    ----------
    input_dim : int
        Length of the flattened observation vector fed to the network.
        For ``IABEnv`` this equals:
        ``2 · num_users + 2 · MAX_NODES + 2 (donor) + 3 (scalars)``.
    action_dim : int
        Number of discrete candidate placement positions (actions).
    hidden_dim : int, optional
        Width of both hidden layers in the shared trunk.  Default: 128.
    device : torch.device or None, optional
        Target compute device.  ``None`` auto-selects via ``get_device()``.
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()

        self.input_dim: int = input_dim
        self.action_dim: int = action_dim
        self.hidden_dim: int = hidden_dim
        self.device: torch.device = device if device is not None else get_device()

        # ── Shared feature trunk ─────────────────────────────────────────
        self.feature_layer: nn.Sequential = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # ── Value stream: scalar state value V(s) ───────────────────────
        self.value_stream: nn.Sequential = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # ── Advantage stream: per-action advantage A(s, a) ──────────────
        self.advantage_stream: nn.Sequential = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        self.to(self.device)

    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-values for all actions given a batch of observations.

        The duelling aggregation follows Wang et al. (2016):

            Q(s, a) = V(s)  +  [ A(s, a) − (1/|A|) Σ_a' A(s, a') ]

        Mean-centering the advantage keeps the decomposition identifiable
        without disrupting the gradient flow.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, input_dim)
            Batch of flattened environment observations.  The tensor must
            already reside on ``self.device``.

        Returns
        -------
        torch.Tensor, shape (batch, action_dim)
            Q-value estimates for every discrete action.
        """
        features: torch.Tensor = self.feature_layer(x)

        value: torch.Tensor = self.value_stream(features)          # (B, 1)
        advantage: torch.Tensor = self.advantage_stream(features)  # (B, A)

        # Mean-centred advantage aggregation
        q_values: torch.Tensor = (
            value + advantage - advantage.mean(dim=1, keepdim=True)
        )
        return q_values

    # ------------------------------------------------------------------ #

    def act(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Select an action using an epsilon-greedy policy.

        With probability ``epsilon`` a uniformly random action is returned
        (exploration); otherwise the greedy action ``argmax Q(s, ·)`` is
        chosen (exploitation).

        Parameters
        ----------
        state : np.ndarray, shape (input_dim,)
            Flattened observation vector from the environment.
        epsilon : float, optional
            Exploration probability in [0, 1].  Default: 0.0 (pure greedy).

        Returns
        -------
        int
            Index of the selected action.
        """
        if random.random() < epsilon:
            return random.randrange(self.action_dim)

        state_tensor: torch.Tensor = (
            torch.tensor(state, dtype=torch.float32)
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():
            q_values: torch.Tensor = self.forward(state_tensor)

        return int(q_values.argmax(dim=1).item())


# ══════════════════════════════════════════════════════════════════════════════
#  Replay Buffer
# ══════════════════════════════════════════════════════════════════════════════

class ReplayBuffer:
    """
    Fixed-capacity experience replay buffer backed by a ``collections.deque``.

    Stores ``Transition`` named-tuples and returns uniformly sampled
    mini-batches as stacked PyTorch tensors, ready for loss computation on
    the target device.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions to retain.  Once full, the oldest
        experience is evicted on each new ``push``.
    device : torch.device or None, optional
        Device onto which sampled tensors are moved.  ``None`` auto-selects
        via ``get_device()``.
    """

    def __init__(
        self,
        capacity: int,
        device: torch.device | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError(
                f"Replay buffer capacity must be positive, got {capacity}."
            )

        self.capacity: int = capacity
        self.device: torch.device = device if device is not None else get_device()
        self._buffer: Deque[Transition] = collections.deque(maxlen=capacity)

    # ------------------------------------------------------------------ #

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Store a single transition in the buffer.

        If the buffer is at capacity, the oldest transition is silently
        dropped (standard circular-buffer behaviour from ``deque.maxlen``).

        Parameters
        ----------
        state : np.ndarray
            Flattened observation before the action was taken.
        action : int
            Discrete action index selected by the agent.
        reward : float
            Scalar reward received from the environment.
        next_state : np.ndarray
            Flattened observation after the action was taken.
        done : bool
            ``True`` if the transition ended the episode.
        """
        self._buffer.append(
            Transition(
                state=np.asarray(state, dtype=np.float32),
                action=int(action),
                reward=float(reward),
                next_state=np.asarray(next_state, dtype=np.float32),
                done=bool(done),
            )
        )

    # ------------------------------------------------------------------ #

    def sample(
        self,
        batch_size: int,
    ) -> Tuple[
        torch.Tensor,  # states
        torch.Tensor,  # actions
        torch.Tensor,  # rewards
        torch.Tensor,  # next_states
        torch.Tensor,  # dones
    ]:
        """
        Draw a uniform random mini-batch of transitions.

        Each field is stacked into a dedicated tensor and moved to
        ``self.device``, matching the format expected by the D3QN loss
        computation:

        - ``states``      : float32, shape (batch_size, input_dim)
        - ``actions``     : int64,   shape (batch_size, 1)
        - ``rewards``     : float32, shape (batch_size, 1)
        - ``next_states`` : float32, shape (batch_size, input_dim)
        - ``dones``       : float32, shape (batch_size, 1)  — 0.0 or 1.0

        Parameters
        ----------
        batch_size : int
            Number of transitions to sample.  Must not exceed ``len(self)``.

        Returns
        -------
        tuple of torch.Tensor
            ``(states, actions, rewards, next_states, dones)``

        Raises
        ------
        ValueError
            If ``batch_size`` exceeds the number of stored transitions.
        """
        if batch_size > len(self):
            raise ValueError(
                f"batch_size ({batch_size}) exceeds buffer size ({len(self)})."
            )

        batch: list[Transition] = random.sample(self._buffer, batch_size)

        states = torch.tensor(
            np.stack([t.state for t in batch]), dtype=torch.float32
        ).to(self.device)

        actions = torch.tensor(
            [[t.action] for t in batch], dtype=torch.int64
        ).to(self.device)

        rewards = torch.tensor(
            [[t.reward] for t in batch], dtype=torch.float32
        ).to(self.device)

        next_states = torch.tensor(
            np.stack([t.next_state for t in batch]), dtype=torch.float32
        ).to(self.device)

        dones = torch.tensor(
            [[float(t.done)] for t in batch], dtype=torch.float32
        ).to(self.device)

        return states, actions, rewards, next_states, dones

    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        """Return the current number of stored transitions."""
        return len(self._buffer)

    def __repr__(self) -> str:
        return (
            f"ReplayBuffer(capacity={self.capacity}, "
            f"stored={len(self)}, device={self.device})"
        )
