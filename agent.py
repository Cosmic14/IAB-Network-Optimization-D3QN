"""
agent.py

D3QN (Duelling Double Deep Q-Network) agent for mmWave IAB small-cell
access point placement optimisation.

Combines:
  - Duelling network architecture  (Wang et al., 2016)
  - Double DQN target computation  (van Hasselt et al., 2016)
  - Polyak-averaged target network (Mnih et al., 2015)
  - Experience replay              (Mnih et al., 2015)
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from environment import IABEnv
from network import DuelingQNetwork, ReplayBuffer, get_device


class D3QNAgent:
    """
    Duelling Double DQN agent that learns to place relay IABNodes on a
    discretised action grid.

    Action space discretisation
    ---------------------------
    The continuous (x, y) placement space is divided into an ``n_bins × n_bins``
    grid of equal-area cells.  Action index ``k`` maps to the centre of cell
    ``(row i, col j)`` where ``k = i · n_bins + j``:

        x_k = (j + 0.5) · grid_width  / n_bins
        y_k = (i + 0.5) · grid_height / n_bins

    This gives ``action_dim = n_bins²`` discrete actions.

    State representation
    --------------------
    The ``IABEnv`` state dict is flattened to a 1-D float32 vector in the
    order:

        [user_positions (N_u·2)] [node_positions (MAX_NODES·2)]
        [donor_position (2)] [num_nodes (1)] [sum_rate_mbps (1)]
        [coverage_rate  (1)]

    Total length: ``2·num_users + 2·MAX_NODES + 5``.

    Double DQN training step
    ------------------------
    For a sampled transition (s, a, r, s', done):

        next_a*   = argmax_a  Q_policy(s')          — policy net selects action
        y         = r  +  γ(1 − done) · Q_target(s', next_a*)  — target net evaluates
        loss      = Huber( Q_policy(s, a),  y.detach() )

    Using the policy net for action selection and the target net for
    evaluation decouples the maximisation bias that afflicts vanilla DQN.

    Parameters
    ----------
    grid_width : float
        Horizontal extent of the ``CityGrid`` [m].
    grid_height : float
        Vertical extent of the ``CityGrid`` [m].
    num_users : int
        Number of users in the environment (determines ``input_dim``).
    n_bins : int
        Number of divisions along each axis.  ``action_dim = n_bins²``.
    buffer_capacity : int
        Maximum transitions stored in the replay buffer.
    lr : float, optional
        Adam learning rate.  Default: 1e-3.
    hidden_dim : int, optional
        Width of both hidden layers in each network.  Default: 128.
    device : torch.device or None, optional
        Compute device.  ``None`` auto-selects via ``get_device()``.
    """

    def __init__(
        self,
        grid_width: float,
        grid_height: float,
        num_users: int,
        n_bins: int,
        buffer_capacity: int,
        lr: float = 1e-3,
        hidden_dim: int = 128,
        device: torch.device | None = None,
    ) -> None:
        self.grid_width: float = grid_width
        self.grid_height: float = grid_height
        self.n_bins: int = n_bins
        self.action_dim: int = n_bins * n_bins
        self.device: torch.device = device if device is not None else get_device()

        # Flattened state dimension derived from IABEnv._build_state schema:
        #   user_positions  : num_users × 2
        #   node_positions  : MAX_NODES × 2
        #   donor_position  : 2
        #   num_nodes       : 1
        #   sum_rate_mbps   : 1
        #   coverage_rate   : 1
        self.input_dim: int = (
            num_users * 2
            + IABEnv.MAX_NODES * 2
            + 2           # donor_position
            + 3           # num_nodes, sum_rate_mbps, coverage_rate
        )

        # ── Networks ─────────────────────────────────────────────────────
        self.policy_net: DuelingQNetwork = DuelingQNetwork(
            input_dim=self.input_dim,
            action_dim=self.action_dim,
            hidden_dim=hidden_dim,
            device=self.device,
        )
        self.target_net: DuelingQNetwork = DuelingQNetwork(
            input_dim=self.input_dim,
            action_dim=self.action_dim,
            hidden_dim=hidden_dim,
            device=self.device,
        )

        # Target net starts as an exact copy; weights are frozen from the
        # optimizer — updated only via soft_update().
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        for param in self.target_net.parameters():
            param.requires_grad = False

        # ── Optimizer and loss ───────────────────────────────────────────
        self.optimizer: optim.Adam = optim.Adam(
            self.policy_net.parameters(), lr=lr
        )
        self.loss_fn: nn.SmoothL1Loss = nn.SmoothL1Loss()

        # ── Replay buffer ────────────────────────────────────────────────
        self.replay_buffer: ReplayBuffer = ReplayBuffer(
            capacity=buffer_capacity,
            device=self.device,
        )

    # ------------------------------------------------------------------ #
    #  State utilities                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def flatten_state(state_dict: dict) -> np.ndarray:
        """
        Convert an ``IABEnv`` state dictionary to a flat float32 array.

        Fields are concatenated in the canonical order defined by
        ``IABEnv._build_state``:

            user_positions  → (N_u · 2,)
            node_positions  → (MAX_NODES · 2,)
            donor_position  → (2,)
            num_nodes       → (1,)
            sum_rate_mbps   → (1,)
            coverage_rate   → (1,)

        Parameters
        ----------
        state_dict : dict
            Observation dictionary as returned by ``IABEnv.reset()`` or
            ``IABEnv.step()``.

        Returns
        -------
        np.ndarray, shape (input_dim,), dtype float32
            Flat observation vector ready for network ingestion.
        """
        return np.concatenate([
            state_dict["user_positions"].ravel(),
            state_dict["node_positions"].ravel(),
            state_dict["donor_position"].ravel(),
            np.array([state_dict["num_nodes"]], dtype=np.float32),
            np.array([state_dict["sum_rate_mbps"]], dtype=np.float32),
            np.array([state_dict["coverage_rate"]], dtype=np.float32),
        ]).astype(np.float32)

    def action_to_coords(self, action: int) -> Tuple[float, float]:
        """
        Map a discrete action index to continuous grid coordinates.

        The grid is divided into ``n_bins × n_bins`` equal cells.  The
        returned point is the **centre** of the corresponding cell:

            row i = action // n_bins        (y-axis bin)
            col j = action  % n_bins        (x-axis bin)

            x = (j + 0.5) · grid_width  / n_bins
            y = (i + 0.5) · grid_height / n_bins

        Parameters
        ----------
        action : int
            Discrete action index in ``[0, action_dim)``.

        Returns
        -------
        tuple[float, float]
            (x, y) coordinates [m] of the bin centre.

        Raises
        ------
        ValueError
            If ``action`` is outside ``[0, action_dim)``.
        """
        if not (0 <= action < self.action_dim):
            raise ValueError(
                f"action {action} out of range [0, {self.action_dim})."
            )
        row: int = action // self.n_bins
        col: int = action % self.n_bins
        x: float = (col + 0.5) * self.grid_width / self.n_bins
        y: float = (row + 0.5) * self.grid_height / self.n_bins
        return x, y

    # ------------------------------------------------------------------ #
    #  Action selection                                                    #
    # ------------------------------------------------------------------ #

    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Choose an action using an epsilon-greedy policy.

        With probability ``epsilon`` a random action index is returned
        (exploration); otherwise the policy network's greedy action
        ``argmax_a Q_policy(state, a)`` is used (exploitation).

        Parameters
        ----------
        state : np.ndarray, shape (input_dim,)
            Flattened observation from ``flatten_state()``.
        epsilon : float, optional
            Exploration probability in [0, 1].  Default: 0.0.

        Returns
        -------
        int
            Selected action index in ``[0, action_dim)``.
        """
        return self.policy_net.act(state, epsilon)

    # ------------------------------------------------------------------ #
    #  Training                                                            #
    # ------------------------------------------------------------------ #

    def train_step(self, batch_size: int, gamma: float = 0.99) -> float:
        """
        Perform one gradient update step using Double DQN.

        **Double DQN logic** (van Hasselt et al., 2016):

        1. Sample a mini-batch ``(s, a, r, s', done)`` from the replay buffer.
        2. Select the greedy next action using the **policy** network:

               a* = argmax_a  Q_policy(s')

        3. Evaluate the next-state Q-value using the **target** network:

               Q_next = Q_target(s', a*)

           This decouples action selection from evaluation, reducing the
           positive bias of vanilla DQN.

        4. Compute the Bellman TD target (detached from the computation graph):

               y = r  +  γ · (1 − done) · Q_next

        5. Compute Huber loss between ``Q_policy(s, a)`` and ``y``, back-propagate,
           and update the policy network weights via Adam.

        Parameters
        ----------
        batch_size : int
            Number of transitions to sample from the replay buffer.
            Must be ≤ ``len(self.replay_buffer)``.
        gamma : float, optional
            Discount factor in (0, 1].  Default: 0.99.

        Returns
        -------
        float
            Scalar training loss for this update step.

        Raises
        ------
        ValueError
            Propagated from ``ReplayBuffer.sample`` if the buffer contains
            fewer than ``batch_size`` transitions.
        """
        states, actions, rewards, next_states, dones = (
            self.replay_buffer.sample(batch_size)
        )
        # shapes:
        #   states, next_states : (B, input_dim)  float32
        #   actions             : (B, 1)           int64
        #   rewards, dones      : (B, 1)           float32

        # ── Current Q-values from policy net ─────────────────────────────
        # Q_policy(s, a)  →  gather the Q-value of the taken action
        q_current: torch.Tensor = (
            self.policy_net(states).gather(dim=1, index=actions)
        )  # (B, 1)

        # ── Double DQN target ─────────────────────────────────────────────
        with torch.no_grad():
            # Step 1: policy net selects the best next action
            next_actions: torch.Tensor = (
                self.policy_net(next_states).argmax(dim=1, keepdim=True)
            )  # (B, 1)  int64 from argmax

            # Step 2: target net evaluates Q at that action
            q_next: torch.Tensor = (
                self.target_net(next_states).gather(dim=1, index=next_actions)
            )  # (B, 1)

            # Bellman target — zero-out future return on terminal transitions
            q_target: torch.Tensor = rewards + gamma * (1.0 - dones) * q_next
            # shape: (B, 1)

        # ── Loss and gradient step ────────────────────────────────────────
        loss: torch.Tensor = self.loss_fn(q_current, q_target)

        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping guards against exploding gradients in early training
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)

        self.optimizer.step()

        return float(loss.item())

    # ------------------------------------------------------------------ #
    #  Target network update                                               #
    # ------------------------------------------------------------------ #

    def soft_update(self, tau: float = 0.005) -> None:
        """
        Polyak-average the target network weights towards the policy network.

        The update rule for each parameter θ_target is:

            θ_target  ←  τ · θ_policy  +  (1 − τ) · θ_target

        Small ``tau`` (e.g., 0.005) ensures the target network moves slowly,
        providing a stable regression target that reduces training oscillation.
        ``tau = 1.0`` is equivalent to a hard copy.

        Parameters
        ----------
        tau : float, optional
            Interpolation coefficient in (0, 1].  Default: 0.005.

        Raises
        ------
        ValueError
            If ``tau`` is outside (0, 1].
        """
        if not (0.0 < tau <= 1.0):
            raise ValueError(f"tau must be in (0, 1], got {tau}.")

        with torch.no_grad():
            for target_param, policy_param in zip(
                self.target_net.parameters(),
                self.policy_net.parameters(),
            ):
                target_param.data.copy_(
                    tau * policy_param.data + (1.0 - tau) * target_param.data
                )
