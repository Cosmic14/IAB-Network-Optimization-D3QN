"""
environment.py

Gym-style RL environment for mmWave IAB small-cell access point placement.
The agent places relay IABNodes sequentially; the environment evaluates the
resulting network using the 60 GHz channel model and returns a reward signal
that balances sum-rate throughput against CAPEX and backhaul feasibility.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from channel_model import ChannelModel
from entities import CityGrid, IABNode, User


class IABEnv:
    """
    Reinforcement Learning environment for IAB relay node placement at 60 GHz.

    The simulation map is a rectangular CityGrid populated with uniformly
    distributed User terminals.  A single **donor** IABNode is fixed at the
    grid centre (fibre-connected, unlimited backhaul).  At each timestep the
    agent places one relay IABNode at continuous (x, y) coordinates.  The
    environment then:

    1. Computes user-to-node and relay-to-donor distance matrices with
       ``np.linalg.norm`` (fully vectorised).
    2. Evaluates stochastic LoS/NLoS pathloss for every pair using the
       3GPP TR 38.901 UMi model embedded in ``ChannelModel`` constants.
    3. Associates each user to the node offering the highest received SNR.
    4. Updates each relay's ``flow_in_capacity`` (backhaul Shannon capacity)
       and ``flow_out_demand`` (sum of associated user demands).
    5. Returns a reward:

           R = ΣC_u  −  100 · Σ𝟙[backhaul violated]  −  50 · |relays|

       where ``C_u`` is the per-user Shannon capacity [Mbps].

    Episode termination occurs when every user's served capacity meets its
    demand, or the relay budget ``MAX_NODES`` is exhausted.

    The state dictionary uses zero-padded, fixed-size arrays so that the
    D3QN observation tensor has a consistent shape across all timesteps.

    Class Constants
    ---------------
    BANDWIDTH_HZ : float
        Access-link channel bandwidth [Hz].  Default: 100 MHz.
    TX_POWER_DBM : float
        Transmit power of every IABNode [dBm].  Default: 23 dBm.
    NOISE_FIGURE_DB : float
        Receiver noise figure [dB].  Default: 7 dB.
    MAX_NODES : int
        Maximum number of relay nodes the agent may deploy per episode.
    CAPEX_PENALTY : float
        Reward deduction per deployed relay node [Mbps-equivalent].
    BACKHAUL_PENALTY : float
        Reward deduction per relay node that violates its backhaul
        capacity constraint [Mbps-equivalent].
    DONOR_FLOW_CAPACITY_MBPS : float
        Fibre backhaul capacity assigned to the donor node [Mbps].
    """

    BANDWIDTH_HZ: float = 100e6
    TX_POWER_DBM: float = 23.0
    NOISE_FIGURE_DB: float = 7.0
    MAX_NODES: int = 10
    CAPEX_PENALTY: float = 50.0
    BACKHAUL_PENALTY: float = 100.0
    DONOR_FLOW_CAPACITY_MBPS: float = 10_000.0

    def __init__(
        self,
        num_users: int,
        seed: int | None = None,
    ) -> None:
        """
        Initialise the IAB environment.

        The simulation area is fixed at a 1000 m × 1000 m grid with 10 m
        resolution, consistent with Zhang et al. (2023) mmWave deployment
        scenarios.

        Parameters
        ----------
        num_users : int
            Number of User terminals to generate per episode.
        seed : int or None, optional
            Master seed for the environment's NumPy random generator.
            Pass an integer for fully reproducible episodes; ``None`` (default)
            draws from OS entropy.
        """
        # Fixed real-world deployment parameters (Zhang et al., 2023)
        self.grid_size: int = 1000        # simulation area side length [m]
        self.resolution: int = 10         # candidate-site spacing [m]
        self.access_radius: int = 200     # max user-to-node access distance [m]
        self.backhaul_radius: int = 300   # max relay-to-donor backhaul distance [m]

        # Discrete candidate deployment sites on the 1000 m grid
        _coords = np.arange(0, self.grid_size, self.resolution)
        self.potential_sites: List[Tuple[int, int]] = [
            (int(x), int(y)) for x in _coords for y in _coords
        ]

        self.grid: CityGrid = CityGrid(
            width=float(self.grid_size), height=float(self.grid_size)
        )
        self.channel: ChannelModel = ChannelModel()
        self.num_users: int = num_users
        self._rng: np.random.Generator = np.random.default_rng(seed)

        # Pre-compute effective noise power once; reused every step.
        # N_eff = N_0 [dBm/Hz] + 10·log10(B) + NF [dB]
        self._noise_power_dbm: float = (
            ChannelModel.NOISE_FLOOR_DBM
            + 10.0 * np.log10(self.BANDWIDTH_HZ)
            + self.NOISE_FIGURE_DB
        )

        # These are populated by reset(); declared here for type clarity.
        self._donors: List[IABNode] = []
        self._donor: IABNode          # alias for _donors[0], set in reset()
        self._relay_nodes: List[IABNode] = []

        self.reset()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def reset(self) -> dict:
        """
        Reset the environment to a clean initial state.

        Clears all relay nodes, regenerates User positions on the CityGrid,
        and re-places the donor node at the grid centre.

        Returns
        -------
        dict
            Initial state observation with zero-padded node positions,
            freshly sampled user positions, and zeroed performance metrics.
            See ``_build_state`` for the full schema.
        """
        self._relay_nodes = []

        # Draw a new user layout using the environment's own RNG.
        user_seed: int = int(self._rng.integers(0, 2**31))
        self.grid.generate_users(self.num_users, seed=user_seed)

        # Spawn 5–20 donor nodes at randomly chosen discrete candidate sites.
        self.num_donors: int = int(self._rng.integers(5, 21))
        site_indices = self._rng.choice(
            len(self.potential_sites), size=self.num_donors, replace=False
        )
        self._donors = [
            IABNode(
                x=float(self.potential_sites[idx][0]),
                y=float(self.potential_sites[idx][1]),
                is_donor=True,
                flow_in_capacity=self.DONOR_FLOW_CAPACITY_MBPS,
                flow_out_demand=0.0,
            )
            for idx in site_indices
        ]
        self._donor = self._donors[0]  # backward-compat alias

        return self._build_state(sum_rate_mbps=0.0, coverage_rate=0.0)

    def step(
        self,
        action_coords: tuple[float, float],
    ) -> tuple[dict, float, bool]:
        """
        Execute one environment step by placing a relay node at ``action_coords``.

        The method performs the following operations in sequence:

        1. **Node placement** — clamp coordinates to the valid grid area and
           append a new relay ``IABNode``.
        2. **Distance matrices** — compute user-to-all-nodes distances
           ``D_ua ∈ ℝ^{N_u × N_a}`` and relay-to-donor distances
           ``D_rd ∈ ℝ^{N_r}`` using vectorised ``np.linalg.norm``.
        3. **SNR matrix** — evaluate stochastic LoS/NLoS pathloss for every
           user-node pair via the 3GPP UMi model (see ``_compute_snr_matrix``).
        4. **User association** — assign each user to the node with the
           highest SNR: ``k*(u) = argmax_k SNR(u, k)``.
        5. **Backhaul update** — for each relay node *i*:

               flow_in_i  = C( SNR(relay_i → donor) )        [Mbps]
               flow_out_i = Σ_{u: k*(u)=i} demand_u          [Mbps]

        6. **Reward computation**::

               R = Σ_u C(SNR_u)
                   − BACKHAUL_PENALTY · Σ_i 𝟙[¬backhaul_ok_i]
                   − CAPEX_PENALTY    · |relay nodes|

        7. **Termination** — ``done = True`` if every user's Shannon capacity
           meets their demand, OR the relay count reaches ``MAX_NODES``.

        Parameters
        ----------
        action_coords : tuple[float, float]
            (x, y) coordinates [m] at which to place the new relay node.
            Values outside the grid are silently clamped to the boundary.

        Returns
        -------
        next_state : dict
            Updated state observation (same schema as ``reset``).
        reward : float
            Scalar reward for this transition.
        done : bool
            ``True`` if the episode has terminated.
        """
        # ── 1. Place relay node ──────────────────────────────────────────
        # Clamp to [0, grid_size) — 1000 m boundary for the Zhang et al. grid.
        x = float(np.clip(action_coords[0], 0.0,
                          np.nextafter(float(self.grid_size), 0.0)))
        y = float(np.clip(action_coords[1], 0.0,
                          np.nextafter(float(self.grid_size), 0.0)))

        new_relay = IABNode(
            x=x, y=y,
            is_donor=False,
            flow_in_capacity=0.0,
            flow_out_demand=0.0,
        )
        self._relay_nodes.append(new_relay)

        # ── 2. Build position arrays ─────────────────────────────────────
        # all_nodes = [donor_0, …, donor_{N_d-1}, relay_0, …, relay_{N_r-1}]
        n_donors: int = len(self._donors)
        all_nodes: List[IABNode] = self._donors + self._relay_nodes

        user_pos: np.ndarray = np.array(
            [[u.x, u.y] for u in self.grid.users], dtype=float
        )  # (N_u, 2)

        node_pos: np.ndarray = np.array(
            [[n.x, n.y] for n in all_nodes], dtype=float
        )  # (N_a, 2)   N_a = N_d + N_r

        donor_pos: np.ndarray = np.array(
            [[d.x, d.y] for d in self._donors], dtype=float
        )  # (N_d, 2)

        relay_pos: np.ndarray = np.array(
            [[n.x, n.y] for n in self._relay_nodes], dtype=float
        )  # (N_r, 2)

        # ── 3. Distance matrices (vectorised) ────────────────────────────
        # D_ua[u, k] = ‖pos_u − pos_k‖₂   shape: (N_u, N_a)
        dist_u2n: np.ndarray = np.linalg.norm(
            user_pos[:, np.newaxis, :] - node_pos[np.newaxis, :, :], axis=2
        )

        # D_rd[i] = min_d ‖pos_relay_i − pos_donor_d‖₂   shape: (N_r,)
        # Each relay backhauled to its nearest donor.
        dist_r2d_all: np.ndarray = np.linalg.norm(
            relay_pos[:, np.newaxis, :] - donor_pos[np.newaxis, :, :], axis=2
        )  # (N_r, N_d)
        dist_r2d: np.ndarray = dist_r2d_all.min(axis=1)  # (N_r,)

        # ── 4. SNR matrix and user association ───────────────────────────
        snr_matrix: np.ndarray = self._compute_snr_matrix(dist_u2n)
        # shape: (N_u, N_a)

        best_node_idx: np.ndarray = np.argmax(snr_matrix, axis=1)   # (N_u,)
        best_snr: np.ndarray = snr_matrix[
            np.arange(self.num_users), best_node_idx
        ]  # (N_u,)

        # Per-user Shannon capacity [Mbps]
        user_capacities: np.ndarray = np.array([
            self.channel.calculate_shannon_capacity(float(snr), self.BANDWIDTH_HZ)
            for snr in best_snr
        ])  # (N_u,)

        sum_rate_mbps: float = float(np.sum(user_capacities))

        # ── 5. Update relay backhaul and access flow ─────────────────────
        user_demands: np.ndarray = np.array(
            [u.data_demand_mbps for u in self.grid.users], dtype=float
        )  # (N_u,)

        for i, relay in enumerate(self._relay_nodes):
            relay_global_idx: int = n_donors + i  # donors occupy indices 0..n_donors-1

            # Backhaul capacity: Shannon capacity of relay → nearest donor link
            d_bh: float = float(dist_r2d[i])
            p_los_bh: float = self.channel.calculate_los_prob(d_bh)
            is_los_bh: bool = bool(self._rng.random() < p_los_bh)
            pl_bh: float = self.channel.calculate_pathloss(d_bh, is_los_bh)
            snr_bh: float = self.channel.calculate_snr(
                self.TX_POWER_DBM, pl_bh, self._noise_power_dbm
            )
            relay.flow_in_capacity = self.channel.calculate_shannon_capacity(
                snr_bh, self.BANDWIDTH_HZ
            )

            # Access demand: aggregate demand of users associated to this relay
            user_mask: np.ndarray = best_node_idx == relay_global_idx
            relay.flow_out_demand = float(np.sum(user_demands[user_mask]))

        # Update each donor's outgoing demand (users directly served by it).
        for d_idx, donor in enumerate(self._donors):
            donor_mask: np.ndarray = best_node_idx == d_idx
            donor.flow_out_demand = float(np.sum(user_demands[donor_mask]))

        # ── 6. Reward ─────────────────────────────────────────────────────
        reward: float = sum_rate_mbps

        for relay in self._relay_nodes:
            if not relay.check_backhaul_constraint():
                reward -= self.BACKHAUL_PENALTY

        reward -= self.CAPEX_PENALTY * len(self._relay_nodes)

        # ── 7. Termination ────────────────────────────────────────────────
        all_demands_met: bool = bool(np.all(user_capacities >= user_demands))
        max_nodes_reached: bool = len(self._relay_nodes) >= self.MAX_NODES
        done: bool = all_demands_met or max_nodes_reached

        coverage_rate: float = float(
            np.mean(user_capacities >= user_demands)
        )

        next_state = self._build_state(
            sum_rate_mbps=sum_rate_mbps,
            coverage_rate=coverage_rate,
        )

        return next_state, float(reward), done

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _compute_snr_matrix(self, dist_matrix: np.ndarray) -> np.ndarray:
        """
        Compute the SNR for every (user, node) pair in a single vectorised pass.

        Replicates the 3GPP TR 38.901 UMi physics of ``ChannelModel`` using
        NumPy array operations to avoid Python-level loops over all pairs.
        The formulas are identical to ``ChannelModel.calculate_los_prob``,
        ``calculate_pathloss``, and ``calculate_snr``; they are inlined here
        for vectorisation efficiency.

        **LoS probability** (3GPP TR 38.901 UMi):

            P_LoS = min(18/d, 1) · (1 − exp(−d/36)) + exp(−d/36)

        **LoS determination**: stochastic draw per pair against P_LoS.

        **Pathloss** [dB]:

            PL_LoS  = 32.4 + 21·log10(d) + 20·log10(f_c)
            PL_NLoS = max(PL_LoS, 35.3·log10(d) + 22.4 + 21.3·log10(f_c))
            A_atm   = (12.6 + 16.0) · d/1000

        **SNR** [linear]:

            SNR = 10^{(P_tx − PL_total − N_eff) / 10}

        Parameters
        ----------
        dist_matrix : np.ndarray, shape (N_u, N_a)
            Euclidean distances between every user and every node [m].

        Returns
        -------
        np.ndarray, shape (N_u, N_a)
            Linear SNR for each (user, node) pair.
        """
        d: np.ndarray = np.maximum(dist_matrix, 1.0)

        # LoS probability
        p_los: np.ndarray = (
            np.minimum(18.0 / d, 1.0) * (1.0 - np.exp(-d / 36.0))
            + np.exp(-d / 36.0)
        )
        p_los = np.clip(p_los, 0.0, 1.0)

        # Stochastic LoS mask
        is_los: np.ndarray = self._rng.random(d.shape) < p_los

        # Base pathloss components
        fc: float = ChannelModel.CARRIER_FREQ_GHZ
        pl_los: np.ndarray = 32.4 + 21.0 * np.log10(d) + 20.0 * np.log10(fc)
        pl_nlos: np.ndarray = np.maximum(
            pl_los,
            35.3 * np.log10(d) + 22.4 + 21.3 * np.log10(fc),
        )

        pl_base: np.ndarray = np.where(is_los, pl_los, pl_nlos)

        # Atmospheric attenuation [dB]
        atm_db_per_m: float = (
            ChannelModel.RAIN_ATTENUATION_DB_PER_KM
            + ChannelModel.OXYGEN_ABSORPTION_DB_PER_KM
        ) / 1000.0
        pl_total: np.ndarray = pl_base + atm_db_per_m * d

        # Linear SNR
        snr_db: np.ndarray = self.TX_POWER_DBM - pl_total - self._noise_power_dbm
        snr_linear: np.ndarray = 10.0 ** (snr_db / 10.0)

        return snr_linear

    def _build_state(
        self,
        sum_rate_mbps: float,
        coverage_rate: float,
    ) -> dict:
        """
        Construct a fixed-schema state observation dictionary.

        Node positions are zero-padded to ``MAX_NODES`` rows so that the
        observation tensor has a consistent shape regardless of how many
        relays have been deployed.  The D3QN policy network can use
        ``num_nodes`` to mask padding entries.

        Parameters
        ----------
        sum_rate_mbps : float
            Current total network throughput [Mbps].
        coverage_rate : float
            Fraction of users whose served capacity meets their demand [0, 1].

        Returns
        -------
        dict with keys:
            ``user_positions``  : np.ndarray (N_u, 2)   — user (x, y) coords
            ``node_positions``  : np.ndarray (MAX_NODES, 2) — relay positions,
                                  zero-padded beyond ``num_nodes``
            ``donor_position``  : np.ndarray (2,)        — donor (x, y)
            ``num_nodes``       : int                    — deployed relay count
            ``sum_rate_mbps``   : float                  — network sum-rate
            ``coverage_rate``   : float                  — user coverage fraction
        """
        user_positions: np.ndarray = np.array(
            [[u.x, u.y] for u in self.grid.users], dtype=float
        )

        # Zero-pad relay positions to a fixed (MAX_NODES, 2) array.
        node_positions: np.ndarray = np.zeros((self.MAX_NODES, 2), dtype=float)
        for i, relay in enumerate(self._relay_nodes):
            node_positions[i] = [relay.x, relay.y]

        donor_positions: np.ndarray = np.array(
            [[d.x, d.y] for d in self._donors], dtype=float
        )  # (N_d, 2) — all active donors

        return {
            "user_positions": user_positions,
            "node_positions": node_positions,
            "donor_position": np.array(
                [self._donor.x, self._donor.y], dtype=float
            ),  # alias: first donor, kept for downstream compat
            "donor_positions": donor_positions,
            "num_nodes": len(self._relay_nodes),
            "sum_rate_mbps": sum_rate_mbps,
            "coverage_rate": coverage_rate,
        }
