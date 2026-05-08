"""
baseline_greedy.py

Static greedy heuristic baseline for IAB relay node placement.
No DRL agent is used. IABEnv is used for environment initialisation and
final evaluation. Saves the deployment map as greedy_baseline.png.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from channel_model import ChannelModel
from entities import IABNode
from environment import IABEnv

# ── Config ─────────────────────────────────────────────────────────────────────
SEED = 42
NUM_USERS = 50
CAPACITY_OVERHEAD = 1.2   # enforced before any link is finalised

# ── Helpers ────────────────────────────────────────────────────────────────────

def dist2d(ax, ay, bx, by):
    return float(np.hypot(ax - bx, ay - by))


def link_capacity(channel, noise_dbm, d_m, rng):
    """Shannon capacity [Mbps] of a single wireless link of distance d_m."""
    p_los = channel.calculate_los_prob(d_m)
    is_los = bool(rng.random() < p_los)
    pl = channel.calculate_pathloss(d_m, is_los)
    snr = channel.calculate_snr(IABEnv.TX_POWER_DBM, pl, noise_dbm)
    return channel.calculate_shannon_capacity(snr, IABEnv.BANDWIDTH_HZ)


# ── Greedy placement ───────────────────────────────────────────────────────────

def run_greedy(env, rng):
    """
    Greedily place relay nodes until all users are served or MAX_NODES is reached.

    Algorithm per outer iteration
    ──────────────────────────────
    1. Sort unconnected users by distance to their nearest existing node.
    2. For each unconnected user (closest-first):
       a. Scan nodes within access_radius (200 m); pick the absolute closest.
          Only finalise the link if capacity >= 1.2 * cumulative demand.
       b. If no node is reachable, place ONE relay at the Euclidean midpoint
          toward the nearest Donor, provided:
            - relay-to-donor distance <= backhaul_radius (300 m)
            - relay backhaul capacity >= 1.2 * user demand

    Returns list of (x, y) relay coordinates in placement order.
    """
    channel = env.channel
    noise_dbm = env._noise_power_dbm
    users = env.grid.users
    donors = list(env._donors)

    # Running node pool: donors first, then greedy-placed relays.
    all_nodes = donors[:]
    node_bh_cap = [d.capacity for d in donors]          # donors: 15 000 Mbps fibre
    node_demand = [0.0] * len(all_nodes)                # cumulative demand routed

    relay_positions = []
    user_connected = [False] * len(users)

    for _ in range(env.MAX_NODES + 1):
        unconn = [i for i, c in enumerate(user_connected) if not c]
        if not unconn:
            break

        # Sort: serve users that are closest to an existing node first.
        unconn.sort(
            key=lambda i: min(dist2d(users[i].x, users[i].y, n.x, n.y)
                              for n in all_nodes)
        )

        progress = False
        relay_placed_this_round = False

        for u_idx in unconn:
            u = users[u_idx]

            # ── a. Try existing nodes within 200 m ────────────────────────
            in_range = sorted(
                (dist2d(u.x, u.y, n.x, n.y), n_idx, n)
                for n_idx, n in enumerate(all_nodes)
                if dist2d(u.x, u.y, n.x, n.y) <= env.access_radius
            )

            served = False
            for d_acc, n_idx, node in in_range:
                new_demand = node_demand[n_idx] + u.data_demand_mbps
                # Constraint 4: capacity >= 1.2 * demand before finalising
                if node_bh_cap[n_idx] >= CAPACITY_OVERHEAD * new_demand:
                    node_demand[n_idx] = new_demand
                    user_connected[u_idx] = True
                    served = True
                    progress = True
                    break

            if served:
                continue

            # ── b. No reachable node — place a relay at the midpoint ──────
            if relay_placed_this_round or len(relay_positions) >= env.MAX_NODES:
                continue

            nearest_donor = min(
                donors, key=lambda d: dist2d(u.x, u.y, d.x, d.y)
            )

            mx = (u.x + nearest_donor.x) / 2.0
            my = (u.y + nearest_donor.y) / 2.0
            d_bh = dist2d(mx, my, nearest_donor.x, nearest_donor.y)

            # Backhaul radius constraint
            if d_bh > env.backhaul_radius:
                continue

            # Relay backhaul capacity from channel model
            bh_cap = link_capacity(channel, noise_dbm, d_bh, rng)

            # Constraint 4: relay capacity must cover at least this user's demand
            if bh_cap < CAPACITY_OVERHEAD * u.data_demand_mbps:
                continue

            # Commit relay placement
            relay_positions.append((mx, my))
            new_relay = IABNode(
                x=mx, y=my,
                is_donor=False,
                flow_in_capacity=bh_cap,
                flow_out_demand=0.0,
            )
            all_nodes.append(new_relay)
            node_bh_cap.append(bh_cap)
            node_demand.append(0.0)
            relay_placed_this_round = True
            progress = True

            # Immediately try to serve this user via the new relay
            d_acc = dist2d(u.x, u.y, mx, my)
            if d_acc <= env.access_radius:
                demand_after = u.data_demand_mbps
                if bh_cap >= CAPACITY_OVERHEAD * demand_after:
                    node_demand[-1] = demand_after
                    user_connected[u_idx] = True

        if not progress:
            break

    return relay_positions


# ── Final evaluation ───────────────────────────────────────────────────────────

def evaluate(relay_positions, eval_seed=SEED):
    """
    Register greedy relay placements in a fresh IABEnv and compute metrics.
    Returns (eval_env, best_node_idx, user_capacities).
    """
    eval_env = IABEnv(num_users=NUM_USERS, seed=eval_seed)
    for pos in relay_positions:
        eval_env.step(pos)

    # Deterministic association for reproducible final metrics
    all_nodes = eval_env._donors + eval_env._relay_nodes
    user_pos = np.array([[u.x, u.y] for u in eval_env.grid.users], dtype=float)
    node_pos = np.array([[n.x, n.y] for n in all_nodes], dtype=float)

    dist_matrix = np.linalg.norm(
        user_pos[:, np.newaxis, :] - node_pos[np.newaxis, :, :], axis=2
    )

    # Swap in a fixed-seed rng for reproducible SNR draws
    saved_rng = eval_env._rng
    eval_env._rng = np.random.default_rng(eval_seed)
    snr_matrix = eval_env._compute_snr_matrix(dist_matrix)
    eval_env._rng = saved_rng

    best_node_idx = np.argmax(snr_matrix, axis=1)
    best_snr = snr_matrix[np.arange(NUM_USERS), best_node_idx]

    user_caps = np.array([
        eval_env.channel.calculate_shannon_capacity(float(s), IABEnv.BANDWIDTH_HZ)
        for s in best_snr
    ])

    return eval_env, best_node_idx, user_caps


# ── Plot ───────────────────────────────────────────────────────────────────────

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


def save_plot(env, best_node_idx, user_caps, save_path="greedy_baseline.png"):
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
        f"Greedy Heuristic Deployment  |  60 GHz  |  "
        f"{len(relays)} Relays  |  "
        f"{int(np.sum(is_connected))}/{len(users)} Users (Shannon ≥ 100 Mbps)",
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

    # Metrics text box
    n_conn = int(np.sum(is_connected))
    sum_rate = float(np.sum(user_caps))
    capex_p = IABEnv.CAPEX_PENALTY * len(relays)
    bh_viol = sum(1 for r in relays if not r.check_backhaul_constraint())
    bh_p = IABEnv.BACKHAUL_PENALTY * bh_viol

    ax.text(
        0.015, 0.975,
        (f"Method          : Greedy Heuristic\n"
         f"Covered (≥100M) : {n_conn}/{len(users)}\n"
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


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    rng = np.random.default_rng(SEED + 1)

    # 1. Initialise environment (no DRL agent)
    env = IABEnv(num_users=NUM_USERS, seed=SEED)

    # 2. Run greedy heuristic
    relay_positions = run_greedy(env, rng)

    # 3. Evaluate placement in a fresh env instance for official metrics
    eval_env, best_node_idx, user_caps = evaluate(relay_positions, eval_seed=SEED)

    user_demands = np.array(
        [u.data_demand_mbps for u in eval_env.grid.users], dtype=float
    )
    is_covered = user_caps >= user_demands          # Shannon capacity >= demand
    n_covered = int(np.sum(is_covered))
    n_abandoned = NUM_USERS - n_covered
    final_sum_rate = float(np.sum(user_caps))
    nodes_deployed = len(eval_env._relay_nodes)

    # 4. Print final metrics (same format as evaluate_agent.py for direct comparison)
    sep = "─" * 60
    print(f"\n{sep}")
    print("  GREEDY BASELINE EVALUATION RESULTS")
    print(sep)
    print(f"  Metric basis              : Shannon capacity >= demand (100 Mbps)")
    print(sep)
    print(f"  Total CAPEX  (Nodes Deployed)      : {nodes_deployed}")
    print(f"  Total Users Covered                : {n_covered} / {NUM_USERS}")
    print(f"  Abandoned Users                    : {n_abandoned}")
    print(f"  Final Sum-Rate                     : {final_sum_rate:.2f} Mbps")
    print(sep + "\n")

    # 5. Save plot (no plt.show())
    save_plot(eval_env, best_node_idx, user_caps, save_path="greedy_baseline.png")


if __name__ == "__main__":
    main()
