"""
evaluate_agent.py

Evaluate the trained D3QN agent on IABEnv in pure-exploitation mode
(epsilon = 0.0).  No learning steps or experience storage are performed.
Saves the deployment map as d3qn_success.png.
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from channel_model import ChannelModel
from environment import IABEnv
from agent import D3QNAgent

# ── Config ─────────────────────────────────────────────────────────────────────
REQUESTED_NUM_USERS = 50
SEED = 42
CHECKPOINT_CANDIDATES = [
    "models/d3qn_checkpoint.pth",
    "d3qn_checkpoint.pth",
    "models/d3qn_policy.pth",
]
PLOT_PATH = "d3qn_success.png"

# ── Locate checkpoint ──────────────────────────────────────────────────────────

def find_checkpoint() -> str:
    for path in CHECKPOINT_CANDIDATES:
        if os.path.isfile(path):
            return path
    sys.exit("ERROR: No checkpoint found. Searched: " + str(CHECKPOINT_CANDIDATES))


# ── Infer architecture from saved weights ─────────────────────────────────────

def infer_arch(checkpoint: dict) -> tuple[int, int, int, int]:
    """
    Returns (input_dim, hidden_dim, action_dim, n_bins) from the weight
    tensors so the network can be rebuilt identically to training time.
    """
    input_dim = checkpoint["feature_layer.0.weight"].shape[1]
    hidden_dim = checkpoint["feature_layer.0.weight"].shape[0]
    action_dim = checkpoint["advantage_stream.2.weight"].shape[0]
    n_bins = int(round(action_dim ** 0.5))
    assert n_bins * n_bins == action_dim, (
        f"action_dim {action_dim} is not a perfect square — cannot infer n_bins."
    )
    return input_dim, hidden_dim, action_dim, n_bins


# ── Compute num_users encoded in the state vector ─────────────────────────────

def infer_num_users(input_dim: int) -> int:
    # input_dim = 2*num_users + 2*MAX_NODES + 2 (donor_pos) + 3 (scalars)
    return (input_dim - 2 * IABEnv.MAX_NODES - 2 - 3) // 2


# ── Vectorised SNR + association (identical physics to IABEnv) ────────────────

def compute_associations(
    env: IABEnv, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Return (best_node_idx, user_capacities) with a fixed RNG seed."""
    rng = np.random.default_rng(seed)
    channel = env.channel

    all_nodes = env._donors + env._relay_nodes
    user_pos = np.array([[u.x, u.y] for u in env.grid.users], dtype=float)
    node_pos = np.array([[n.x, n.y] for n in all_nodes], dtype=float)

    d = np.maximum(
        np.linalg.norm(
            user_pos[:, np.newaxis, :] - node_pos[np.newaxis, :, :], axis=2
        ),
        1.0,
    )

    p_los = (
        np.minimum(18.0 / d, 1.0) * (1.0 - np.exp(-d / 36.0)) + np.exp(-d / 36.0)
    )
    is_los = rng.random(d.shape) < np.clip(p_los, 0.0, 1.0)

    fc = ChannelModel.CARRIER_FREQ_GHZ
    pl_los = 32.4 + 21.0 * np.log10(d) + 20.0 * np.log10(fc)
    pl_nlos = np.maximum(pl_los, 35.3 * np.log10(d) + 22.4 + 21.3 * np.log10(fc))
    atm = (ChannelModel.RAIN_ATTENUATION_DB_PER_KM + ChannelModel.OXYGEN_ABSORPTION_DB_PER_KM) / 1000.0
    pl_total = np.where(is_los, pl_los, pl_nlos) + atm * d

    noise_dbm = (
        ChannelModel.NOISE_FLOOR_DBM
        + 10.0 * np.log10(IABEnv.BANDWIDTH_HZ)
        + IABEnv.NOISE_FIGURE_DB
    )
    snr = 10.0 ** ((IABEnv.TX_POWER_DBM - pl_total - noise_dbm) / 10.0)

    best_idx = np.argmax(snr, axis=1)
    best_snr = snr[np.arange(len(env.grid.users)), best_idx]

    caps = np.array([
        channel.calculate_shannon_capacity(float(s), IABEnv.BANDWIDTH_HZ)
        for s in best_snr
    ])
    return best_idx, caps


# ── Deployment plot (no plt.show) ──────────────────────────────────────────────

_NODE_COLOURS = [
    "#e41a1c", "#377eb8", "#ff7f00", "#4daf4a",
    "#984ea3", "#a65628", "#f781bf", "#999999",
    "#ffff33", "#e41a1c", "#377eb8",
]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.grid": True,
    "grid.color": "#cccccc",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "figure.dpi": 120,
    "figure.facecolor": "white",
})


def render_deployment(
    env: IABEnv,
    best_node_idx: np.ndarray,
    user_caps: np.ndarray,
    save_path: str,
) -> None:
    """Render the IAB deployment map and save to save_path (no plt.show)."""
    users = env.grid.users
    donors = env._donors
    relays = env._relay_nodes
    all_nodes = donors + relays
    n_donors = len(donors)

    user_demands = np.array([u.data_demand_mbps for u in users])
    is_connected = user_caps >= user_demands

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.set_xlim(-10, 1010)
    ax.set_ylim(-10, 1010)
    ax.set_xlabel("x  [m]")
    ax.set_ylabel("y  [m]")
    ax.set_title(
        f"D3QN Agent Deployment  |  60 GHz  |  "
        f"{len(relays)} Relays  |  "
        f"{int(np.sum(is_connected))}/{len(users)} Users Connected",
        pad=12,
    )
    ax.set_aspect("equal")
    ax.add_patch(plt.Rectangle(
        (0, 0), env.grid.width, env.grid.height,
        linewidth=1.5, edgecolor="#555555", facecolor="#f9f9f9", zorder=0,
    ))

    # Relay → nearest donor backhaul lines
    donor_pos_arr = np.array([[d.x, d.y] for d in donors])
    for i, relay in enumerate(relays):
        colour = _NODE_COLOURS[(n_donors + i) % len(_NODE_COLOURS)]
        dists = np.linalg.norm(donor_pos_arr - [relay.x, relay.y], axis=1)
        nearest = donors[int(np.argmin(dists))]
        ax.plot([relay.x, nearest.x], [relay.y, nearest.y],
                linestyle="--", linewidth=1.8, color=colour, alpha=0.75, zorder=1)

    # User → best node access lines
    for u_idx, user in enumerate(users):
        node = all_nodes[best_node_idx[u_idx]]
        colour = _NODE_COLOURS[best_node_idx[u_idx] % len(_NODE_COLOURS)]
        ax.plot([user.x, node.x], [user.y, node.y],
                linestyle="--", linewidth=0.7, color=colour, alpha=0.4, zorder=2)

    # Users
    ux = np.array([u.x for u in users])
    uy = np.array([u.y for u in users])
    ax.scatter(ux[~is_connected], uy[~is_connected],
               s=70, color="#aaaaaa", edgecolors="#666666",
               linewidths=0.8, zorder=4, label="User (unconnected)")
    ax.scatter(ux[is_connected], uy[is_connected],
               s=70, color="#4daf4a", edgecolors="#2a7a29",
               linewidths=0.8, zorder=4, label="User (connected)")

    # Relay nodes
    for i, relay in enumerate(relays):
        colour = _NODE_COLOURS[(n_donors + i) % len(_NODE_COLOURS)]
        edge = "#000000" if relay.check_backhaul_constraint() else "#ff0000"
        ax.scatter(relay.x, relay.y, s=220, marker="s", color=colour,
                   edgecolors=edge, linewidths=2.0, zorder=6)
        ax.annotate(
            f"R{i+1}\n{relay.flow_in_capacity:.0f}↓\n{relay.flow_out_demand:.0f}↑",
            xy=(relay.x, relay.y), xytext=(10, 6), textcoords="offset points",
            fontsize=8, color=colour, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7), zorder=7,
        )

    # Donor nodes
    for d_idx, donor in enumerate(donors):
        ax.scatter(donor.x, donor.y, s=350, marker="*", color="#e41a1c",
                   edgecolors="#8b0000", linewidths=1.5, zorder=6,
                   label="Donors (fibre)" if d_idx == 0 else "_nolegend_")
        ax.annotate(
            f"D{d_idx+1}", xy=(donor.x, donor.y),
            xytext=(10, 8), textcoords="offset points",
            fontsize=9, color="#e41a1c", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8), zorder=7,
        )

    # Legend
    relay_handles = [
        mlines.Line2D(
            [], [], marker="s", color="w",
            markerfacecolor=_NODE_COLOURS[(n_donors + i) % len(_NODE_COLOURS)],
            markeredgecolor="#000000", markersize=10,
            label=f"Relay {i+1}  ({r.x:.0f}, {r.y:.0f}) m",
        )
        for i, r in enumerate(relays)
    ]
    ax.legend(
        handles=ax.get_legend_handles_labels()[0][:2]
                + relay_handles
                + [
                    mlines.Line2D([], [], color="#377eb8", linestyle="--",
                                  linewidth=1.8, label="Backhaul (relay→donor)"),
                    mlines.Line2D([], [], color="#aaaaaa", linestyle="--",
                                  linewidth=0.8, label="Access (user→node)"),
                ],
        loc="upper left", fontsize=9, framealpha=1.0, edgecolor="#cccccc",
    )

    # Metrics box
    n_conn = int(np.sum(is_connected))
    sum_rate = float(np.sum(user_caps))
    capex_p = IABEnv.CAPEX_PENALTY * len(relays)
    bh_viol = sum(1 for r in relays if not r.check_backhaul_constraint())
    bh_p = IABEnv.BACKHAUL_PENALTY * bh_viol

    ax.text(
        0.015, 0.975,
        (f"Method          : D3QN Agent (ε=0.0)\n"
         f"Users connected : {n_conn}/{len(users)}\n"
         f"Sum-rate        : {sum_rate:.1f} Mbps\n"
         f"Relays deployed : {len(relays)}\n"
         f"CAPEX penalty   : −{capex_p:.0f}\n"
         f"BH violations   : {bh_viol}"),
        transform=ax.transAxes, fontsize=9, verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.88,
                  edgecolor="#aaaaaa"),
        zorder=8,
    )

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Plot saved → {save_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── 1. Load checkpoint and infer architecture ──────────────────────────
    ckpt_path = find_checkpoint()
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    input_dim, hidden_dim, action_dim, n_bins = infer_arch(checkpoint)
    actual_num_users = infer_num_users(input_dim)

    sep = "─" * 60
    print(f"\n{sep}")
    print("  D3QN AGENT EVALUATION")
    print(sep)
    print(f"  Checkpoint        : {ckpt_path}")
    print(f"  Architecture      : input={input_dim}  hidden={hidden_dim}"
          f"  actions={action_dim} ({n_bins}×{n_bins} grid)")

    if actual_num_users != REQUESTED_NUM_USERS:
        print(f"\n  [WARNING] Checkpoint was trained with num_users={actual_num_users}.")
        print(f"            Requested num_users={REQUESTED_NUM_USERS} would produce a")
        print(f"            state dimension of {REQUESTED_NUM_USERS*2 + 2*IABEnv.MAX_NODES + 5},"
              f" incompatible with the loaded network (input_dim={input_dim}).")
        print(f"            Evaluating with num_users={actual_num_users} to match checkpoint.\n")

    num_users = actual_num_users

    # ── 2. Initialise environment ──────────────────────────────────────────
    # seed=42 is preserved; num_users is forced to match the trained checkpoint.
    env = IABEnv(num_users=num_users, seed=SEED)

    # ── 3. Build agent and load weights ───────────────────────────────────
    agent = D3QNAgent(
        grid_width=float(env.grid_size),
        grid_height=float(env.grid_size),
        num_users=num_users,
        n_bins=n_bins,
        buffer_capacity=1,       # minimal; never used in evaluation
        hidden_dim=hidden_dim,
    )
    agent.policy_net.load_state_dict(checkpoint)
    agent.policy_net.eval()      # inference mode — disables dropout / batchnorm

    print(f"  Environment       : IABEnv(num_users={num_users}, seed={SEED})")
    print(f"  Donors            : {len(env._donors)}")
    print(f"  Users             : {num_users}")
    print(sep)

    # ── 4. Run one episode with ε = 0.0 (pure exploitation) ───────────────
    state = env.reset()
    done = False
    episode_reward = 0.0
    steps = 0

    while not done:
        flat_state = D3QNAgent.flatten_state(state)
        action = agent.select_action(flat_state, epsilon=0.0)  # no exploration
        coords = agent.action_to_coords(action)
        state, reward, done = env.step(coords)
        episode_reward += reward
        steps += 1

    # ── 5. Final metrics ───────────────────────────────────────────────────
    best_node_idx, user_caps = compute_associations(env, seed=SEED)

    user_demands = np.array([u.data_demand_mbps for u in env.grid.users])
    is_connected = user_caps >= user_demands
    n_covered = int(np.sum(is_connected))
    n_abandoned = num_users - n_covered
    final_sum_rate = float(np.sum(user_caps))
    nodes_deployed = len(env._relay_nodes)

    print(f"  Episode steps     : {steps}")
    print(f"  Episode reward    : {episode_reward:.2f}")
    print(sep)
    print("  FINAL EVALUATION METRICS")
    print(sep)
    print(f"  Total CAPEX  (New Relays Deployed) : {nodes_deployed}")
    print(f"  Total Users Covered                : {n_covered} / {num_users}")
    print(f"  Abandoned Users                    : {n_abandoned}")
    print(f"  Final Sum-Rate                     : {final_sum_rate:.2f} Mbps")
    print(sep + "\n")

    # ── 6. Render and save deployment plot ─────────────────────────────────
    render_deployment(env, best_node_idx, user_caps, save_path=PLOT_PATH)


if __name__ == "__main__":
    main()
