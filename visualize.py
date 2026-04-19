"""
visualize.py

Deployment visualisation for the mmWave IAB small-cell placement simulation.
Demonstrates a 3-relay IAB network on a 500 × 500 m CityGrid and prints a
full link-budget summary to the terminal.

Usage
-----
    python3 visualize.py
"""

from __future__ import annotations

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

from channel_model import ChannelModel
from environment import IABEnv


# ── Plotting style ────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          11,
    "axes.titlesize":     13,
    "axes.titleweight":   "bold",
    "axes.labelsize":     11,
    "axes.grid":          True,
    "grid.color":         "#cccccc",
    "grid.linestyle":     "--",
    "grid.linewidth":     0.6,
    "figure.dpi":         120,
    "figure.facecolor":   "white",
})

# Per-node colour palette (donor + up to MAX_NODES relays)
_NODE_COLOURS = ["#e41a1c", "#377eb8", "#ff7f00", "#4daf4a",
                 "#984ea3", "#a65628", "#f781bf", "#999999",
                 "#ffff33", "#e41a1c"]


# ── Association helper ────────────────────────────────────────────────────────

def compute_associations(
    env: IABEnv,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Deterministically recompute user-to-node associations for visualisation.

    Uses the same 3GPP TR 38.901 UMi physics as ``IABEnv._compute_snr_matrix``
    but with a fixed ``seed`` so the plot is reproducible regardless of the
    stochastic draws made during training or stepping.

    Parameters
    ----------
    env : IABEnv
        Environment after all relay nodes have been placed via ``step()``.
    seed : int, optional
        Seed for the LoS/NLoS random draw.  Default: 0.

    Returns
    -------
    best_node_idx : np.ndarray, shape (N_u,)
        Index into ``[donor] + relay_nodes`` for each user's best node.
    user_capacities : np.ndarray, shape (N_u,)  [Mbps]
        Shannon capacity served to each user by their best node.
    """
    rng = np.random.default_rng(seed)
    channel = ChannelModel()

    all_nodes = [env._donor] + env._relay_nodes
    user_pos = np.array([[u.x, u.y] for u in env.grid.users], dtype=float)
    node_pos = np.array([[n.x, n.y] for n in all_nodes],       dtype=float)

    # Distance matrix  (N_u, N_a)
    dist = np.linalg.norm(
        user_pos[:, np.newaxis, :] - node_pos[np.newaxis, :, :], axis=2
    )
    d = np.maximum(dist, 1.0)

    # LoS probability and stochastic draw
    p_los = (
        np.minimum(18.0 / d, 1.0) * (1.0 - np.exp(-d / 36.0))
        + np.exp(-d / 36.0)
    )
    is_los = rng.random(d.shape) < np.clip(p_los, 0.0, 1.0)

    # Pathloss  (dB)
    fc = ChannelModel.CARRIER_FREQ_GHZ
    pl_los  = 32.4 + 21.0 * np.log10(d) + 20.0 * np.log10(fc)
    pl_nlos = np.maximum(pl_los,
                         35.3 * np.log10(d) + 22.4 + 21.3 * np.log10(fc))
    atm_db_per_m = (ChannelModel.RAIN_ATTENUATION_DB_PER_KM
                    + ChannelModel.OXYGEN_ABSORPTION_DB_PER_KM) / 1000.0
    pl_total = np.where(is_los, pl_los, pl_nlos) + atm_db_per_m * d

    # SNR  (linear)
    noise_dbm = (ChannelModel.NOISE_FLOOR_DBM
                 + 10.0 * np.log10(IABEnv.BANDWIDTH_HZ)
                 + IABEnv.NOISE_FIGURE_DB)
    snr_linear = 10.0 ** ((IABEnv.TX_POWER_DBM - pl_total - noise_dbm) / 10.0)

    best_node_idx = np.argmax(snr_linear, axis=1)       # (N_u,)
    best_snr      = snr_linear[np.arange(len(env.grid.users)), best_node_idx]

    user_capacities = np.array([
        channel.calculate_shannon_capacity(float(s), IABEnv.BANDWIDTH_HZ)
        for s in best_snr
    ])
    return best_node_idx, user_capacities


# ── Terminal summary ──────────────────────────────────────────────────────────

def print_summary(
    env: IABEnv,
    best_node_idx: np.ndarray,
    user_capacities: np.ndarray,
    step_rewards: list[float],
) -> None:
    """
    Print a structured link-budget and reward summary to stdout.

    Parameters
    ----------
    env : IABEnv
        Environment after all relay placements.
    best_node_idx : np.ndarray
        User-to-node association vector from ``compute_associations``.
    user_capacities : np.ndarray
        Per-user served capacity [Mbps].
    step_rewards : list[float]
        Raw reward returned by each ``env.step()`` call.
    """
    users       = env.grid.users
    n_users     = len(users)
    demands     = np.array([u.data_demand_mbps for u in users])
    connected   = user_capacities >= demands
    n_connected = int(np.sum(connected))
    sum_rate    = float(np.sum(user_capacities))

    relays      = env._relay_nodes
    n_relays    = len(relays)
    violations  = [r for r in relays if not r.check_backhaul_constraint()]
    n_violations = len(violations)

    capex_penalty     = IABEnv.CAPEX_PENALTY    * n_relays
    backhaul_penalty  = IABEnv.BACKHAUL_PENALTY * n_violations
    net_reward        = sum_rate - capex_penalty - backhaul_penalty

    sep = "─" * 56
    print(f"\n{sep}")
    print("  IAB DEPLOYMENT SUMMARY")
    print(sep)
    print(f"  Grid              : {env.grid.width:.0f} m × {env.grid.height:.0f} m")
    print(f"  Carrier frequency : {ChannelModel.CARRIER_FREQ_GHZ:.0f} GHz")
    print(f"  Bandwidth         : {IABEnv.BANDWIDTH_HZ/1e6:.0f} MHz")
    print(f"  Tx power          : {IABEnv.TX_POWER_DBM:.0f} dBm")
    print(sep)
    print(f"  Total users       : {n_users}")
    print(f"  Connected users   : {n_connected}  ({100*n_connected/n_users:.0f} %)")
    print(f"  Unconnected users : {n_users - n_connected}")
    print(sep)
    print(f"  Relay nodes       : {n_relays}")
    for i, relay in enumerate(relays):
        status = "OK" if relay.check_backhaul_constraint() else "VIOLATED"
        print(f"    Relay {i+1}  ({relay.x:6.1f}, {relay.y:6.1f}) m  "
              f"| flow_in={relay.flow_in_capacity:7.2f} Mbps  "
              f"| flow_out={relay.flow_out_demand:7.2f} Mbps  [{status}]")
    print(sep)
    print(f"  Network sum-rate  : {sum_rate:>10.2f} Mbps")
    print(f"  CAPEX penalty     : {-capex_penalty:>10.2f}  "
          f"({n_relays} relays × {IABEnv.CAPEX_PENALTY:.0f})")
    print(f"  Backhaul penalty  : {-backhaul_penalty:>10.2f}  "
          f"({n_violations} violation(s) × {IABEnv.BACKHAUL_PENALTY:.0f})")
    print(f"  Net reward        : {net_reward:>10.2f}")
    print(sep)
    for k, r in enumerate(step_rewards, start=1):
        print(f"  Step {k} raw reward : {r:>10.2f}")
    print(sep + "\n")


# ── Main plot ─────────────────────────────────────────────────────────────────

def visualize() -> None:
    """
    Run a 3-relay IAB deployment and render the 2-D CityGrid plot.

    Steps
    -----
    1. Instantiate ``IABEnv`` with a fixed seed (reproducible user layout).
    2. Place three relay nodes at pre-selected coordinates via ``step()``.
    3. Recompute user associations deterministically for the plot.
    4. Draw: Users (green/grey), Donor (red ★), Relays (blue ■),
       user→node dashed lines, relay→donor dashed lines.
    5. Print terminal summary.
    6. Display the figure.
    """
    # ── 1. Environment setup ──────────────────────────────────────────────
    env = IABEnv(width=500.0, height=500.0, num_users=20, seed=42)
    env.reset()

    # ── 2. Place three relay nodes ────────────────────────────────────────
    relay_coords = [
        (110.0, 110.0),   # bottom-left coverage zone
        (390.0, 110.0),   # bottom-right coverage zone
        (250.0, 400.0),   # top-centre coverage zone
    ]

    step_rewards: list[float] = []
    last_state: dict = {}

    for coords in relay_coords:
        state, reward, done = env.step(coords)
        step_rewards.append(reward)
        last_state = state

    # ── 3. Associations and capacities ───────────────────────────────────
    best_node_idx, user_capacities = compute_associations(env, seed=0)
    all_nodes = [env._donor] + env._relay_nodes

    user_demands = np.array([u.data_demand_mbps for u in env.grid.users])
    is_connected = user_capacities >= user_demands

    # ── 4. Build figure ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.set_xlim(-10, 510)
    ax.set_ylim(-10, 510)
    ax.set_xlabel("x  [m]")
    ax.set_ylabel("y  [m]")
    ax.set_title(
        f"mmWave IAB Deployment  |  60 GHz  |  "
        f"{len(env._relay_nodes)} Relays  |  "
        f"{int(np.sum(is_connected))}/{len(env.grid.users)} Users Connected",
        pad=12,
    )
    ax.set_aspect("equal")

    # Grid boundary rectangle
    boundary = plt.Rectangle(
        (0, 0), env.grid.width, env.grid.height,
        linewidth=1.5, edgecolor="#555555", facecolor="#f9f9f9", zorder=0,
    )
    ax.add_patch(boundary)

    # ── 4a. Connection lines: relay → donor ──────────────────────────────
    donor = env._donor
    for i, relay in enumerate(env._relay_nodes):
        colour = _NODE_COLOURS[i + 1]
        ax.plot(
            [relay.x, donor.x], [relay.y, donor.y],
            linestyle="--", linewidth=1.8, color=colour,
            alpha=0.75, zorder=1,
        )

    # ── 4b. Connection lines: user → best node ───────────────────────────
    for u_idx, user in enumerate(env.grid.users):
        node   = all_nodes[best_node_idx[u_idx]]
        colour = _NODE_COLOURS[best_node_idx[u_idx]]
        ax.plot(
            [user.x, node.x], [user.y, node.y],
            linestyle="--", linewidth=0.8, color=colour,
            alpha=0.45, zorder=2,
        )

    # ── 4c. Users ─────────────────────────────────────────────────────────
    ux = np.array([u.x for u in env.grid.users])
    uy = np.array([u.y for u in env.grid.users])

    ax.scatter(
        ux[~is_connected], uy[~is_connected],
        s=70, color="#aaaaaa", edgecolors="#666666",
        linewidths=0.8, zorder=4, label="User (unconnected)",
    )
    ax.scatter(
        ux[is_connected], uy[is_connected],
        s=70, color="#4daf4a", edgecolors="#2a7a29",
        linewidths=0.8, zorder=4, label="User (connected)",
    )

    # Annotate per-user capacity
    for u_idx, user in enumerate(env.grid.users):
        cap_str = f"{user_capacities[u_idx]:.0f}"
        ax.annotate(
            cap_str,
            xy=(user.x, user.y),
            xytext=(4, 4), textcoords="offset points",
            fontsize=7, color="#333333", zorder=5,
        )

    # ── 4d. Relay nodes ───────────────────────────────────────────────────
    for i, relay in enumerate(env._relay_nodes):
        colour = _NODE_COLOURS[i + 1]
        bh_ok  = relay.check_backhaul_constraint()
        edge   = "#000000" if bh_ok else "#ff0000"
        ax.scatter(
            relay.x, relay.y,
            s=220, marker="s", color=colour,
            edgecolors=edge, linewidths=2.0, zorder=6,
        )
        ax.annotate(
            f"R{i+1}\n{relay.flow_in_capacity:.0f}↓\n{relay.flow_out_demand:.0f}↑",
            xy=(relay.x, relay.y),
            xytext=(10, 6), textcoords="offset points",
            fontsize=8, color=colour, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
            zorder=7,
        )

    # ── 4e. Donor node ────────────────────────────────────────────────────
    ax.scatter(
        donor.x, donor.y,
        s=350, marker="*", color="#e41a1c",
        edgecolors="#8b0000", linewidths=1.5, zorder=6,
        label="Donor (fibre)",
    )
    ax.annotate(
        "Donor",
        xy=(donor.x, donor.y),
        xytext=(10, 8), textcoords="offset points",
        fontsize=9, color="#e41a1c", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8),
        zorder=7,
    )

    # ── 4f. Legend ────────────────────────────────────────────────────────
    relay_handles = [
        mlines.Line2D(
            [], [],
            marker="s", color="w",
            markerfacecolor=_NODE_COLOURS[i + 1],
            markeredgecolor="#000000",
            markersize=10,
            label=f"Relay {i+1}  ({r.x:.0f}, {r.y:.0f}) m",
        )
        for i, r in enumerate(env._relay_nodes)
    ]
    bh_line = mlines.Line2D(
        [], [], color="#377eb8", linestyle="--",
        linewidth=1.8, label="Backhaul link (relay → donor)",
    )
    conn_line = mlines.Line2D(
        [], [], color="#aaaaaa", linestyle="--",
        linewidth=0.8, label="Access link (user → node)",
    )

    ax.legend(
        handles=ax.get_legend_handles_labels()[0][:2]
                + relay_handles + [bh_line, conn_line],
        loc="upper right",
        fontsize=9,
        framealpha=0.9,
        edgecolor="#cccccc",
    )

    # ── 4g. Metrics text box ──────────────────────────────────────────────
    n_conn    = int(np.sum(is_connected))
    sum_rate  = float(np.sum(user_capacities))
    capex_p   = IABEnv.CAPEX_PENALTY * len(env._relay_nodes)
    bh_viol   = sum(1 for r in env._relay_nodes
                    if not r.check_backhaul_constraint())
    bh_p      = IABEnv.BACKHAUL_PENALTY * bh_viol
    net_rew   = sum_rate - capex_p - bh_p

    metrics = (
        f"Users connected : {n_conn}/{len(env.grid.users)}\n"
        f"Sum-rate        : {sum_rate:.1f} Mbps\n"
        f"CAPEX penalty   : −{capex_p:.0f}\n"
        f"BH penalty      : −{bh_p:.0f}\n"
        f"Net reward      : {net_rew:.1f}"
    )
    ax.text(
        0.015, 0.975, metrics,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.88,
                  edgecolor="#aaaaaa"),
        zorder=8,
    )

    plt.tight_layout()

    # ── 5. Terminal summary ───────────────────────────────────────────────
    print_summary(env, best_node_idx, user_capacities, step_rewards)

    # ── 6. Show ───────────────────────────────────────────────────────────
    plt.savefig("iab_deployment.png", bbox_inches="tight", dpi=150)
    print("  Plot saved → iab_deployment.png")
    plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    visualize()
