"""
train.py

Training script for the D3QN IAB small-cell placement agent.

Usage
-----
    python3 train.py
    python3 train.py --episodes 3000 --batch_size 128 --gamma 0.99
    python3 train.py --epsilon_decay 0.998 --n_bins 10

Checkpoints are saved to models/d3qn_policy.pth every 500 episodes and
at the end of training.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from agent import D3QNAgent
from environment import IABEnv


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """
    Parse command-line hyperparameters.

    Returns
    -------
    argparse.Namespace
        Parsed argument namespace with all training hyperparameters.
    """
    parser = argparse.ArgumentParser(
        description="Train a D3QN agent for mmWave IAB small-cell placement."
    )

    # Environment (grid fixed at 1000 m × 1000 m per Zhang et al. 2023)
    parser.add_argument("--num_users",  type=int,   default=10,
                        help="Number of UEs per episode (default: 10)")
    parser.add_argument("--n_bins",     type=int,   default=7,
                        help="Action grid bins per axis; action_dim = n_bins² "
                             "(default: 7 → 49 actions)")

    # Training loop
    parser.add_argument("--episodes",       type=int,   default=2000,
                        help="Total training episodes (default: 2000)")
    parser.add_argument("--batch_size",     type=int,   default=64,
                        help="Replay buffer sample size (default: 64)")
    parser.add_argument("--gamma",          type=float, default=0.99,
                        help="Discount factor (default: 0.99)")
    parser.add_argument("--tau",            type=float, default=0.005,
                        help="Polyak soft-update coefficient (default: 0.005)")
    parser.add_argument("--lr",             type=float, default=1e-3,
                        help="Adam learning rate (default: 1e-3)")
    parser.add_argument("--buffer_capacity",type=int,   default=10_000,
                        help="Replay buffer capacity (default: 10 000)")
    parser.add_argument("--min_buffer",     type=int,   default=500,
                        help="Minimum transitions before training starts "
                             "(default: 500)")
    parser.add_argument("--hidden_dim",     type=int,   default=128,
                        help="Hidden layer width (default: 128)")

    # Epsilon-greedy schedule
    parser.add_argument("--epsilon_start", type=float, default=1.0,
                        help="Initial exploration rate (default: 1.0)")
    parser.add_argument("--epsilon_end",   type=float, default=0.01,
                        help="Minimum exploration rate (default: 0.01)")
    parser.add_argument("--epsilon_decay", type=float, default=0.995,
                        help="Multiplicative decay per episode (default: 0.995)")

    # Checkpointing
    parser.add_argument("--checkpoint_dir",    type=str, default="models",
                        help="Directory for saved weights (default: models/)")
    parser.add_argument("--checkpoint_every",  type=int, default=500,
                        help="Save checkpoint every N episodes (default: 500)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility (default: None)")

    return parser.parse_args()


# ── Checkpoint helper ─────────────────────────────────────────────────────────

def save_checkpoint(
    agent: D3QNAgent,
    checkpoint_dir: Path,
    episode: int,
) -> None:
    """
    Persist the policy network's state dict to disk.

    The file is always written to the same path so that the latest checkpoint
    overwrites the previous one, keeping disk usage bounded.

    Parameters
    ----------
    agent : D3QNAgent
        The agent whose policy network weights are saved.
    checkpoint_dir : Path
        Directory in which to write ``d3qn_policy.pth``.
    episode : int
        Current episode number, included in the console message only.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_path: Path = checkpoint_dir / "d3qn_policy.pth"
    torch.save(agent.policy_net.state_dict(), save_path)
    print(f"  [checkpoint] saved → {save_path}  (episode {episode})")


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    """
    Run the full D3QN training loop.

    Episode flow
    ------------
    1. ``env.reset()``  — fresh user layout, donor at centre, no relays.
    2. Flatten state dict → 1-D numpy observation.
    3. ``agent.select_action(state, epsilon)``  — epsilon-greedy over bins.
    4. ``agent.action_to_coords(action)``       — discrete → (x, y).
    5. ``env.step(coords)``                      — physics + reward.
    6. ``agent.replay_buffer.push(...)``         — store transition.
    7. ``agent.train_step(batch_size, gamma)``   — Double DQN update.
    8. ``agent.soft_update(tau)``                — Polyak target update.
    9. Decay epsilon; log; checkpoint if due.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed hyperparameters from ``parse_args()``.
    """
    # ── Seeding ──────────────────────────────────────────────────────────
    if args.seed is not None:
        import random
        import numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    checkpoint_dir = Path(args.checkpoint_dir)

    # ── Environment and agent ────────────────────────────────────────────
    env = IABEnv(
        num_users=args.num_users,
        seed=args.seed,
    )

    agent = D3QNAgent(
        grid_width=float(env.grid_size),
        grid_height=float(env.grid_size),
        num_users=args.num_users,
        n_bins=args.n_bins,
        buffer_capacity=args.buffer_capacity,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
    )

    epsilon: float = args.epsilon_start

    # ── Header ───────────────────────────────────────────────────────────
    print(
        f"\n{'─'*70}\n"
        f"  D3QN IAB Training\n"
        f"  Episodes      : {args.episodes}\n"
        f"  Action grid   : {args.n_bins}×{args.n_bins} = {args.n_bins**2} actions\n"
        f"  Input dim     : {agent.input_dim}\n"
        f"  Buffer cap    : {args.buffer_capacity:,}\n"
        f"  Device        : {agent.device}\n"
        f"  Seed          : {args.seed}\n"
        f"{'─'*70}\n"
        f"{'Episode':>8} {'Reward':>10} {'SumRate(Mbps)':>14} "
        f"{'Nodes':>6} {'ε':>7} {'Loss':>10}"
    )
    print("─" * 70)

    # ── Main loop ────────────────────────────────────────────────────────
    for episode in range(1, args.episodes + 1):

        state_dict = env.reset()
        state = D3QNAgent.flatten_state(state_dict)

        episode_reward: float = 0.0
        episode_loss: float = 0.0
        loss_steps: int = 0
        done: bool = False

        while not done:
            # ── Action selection ─────────────────────────────────────────
            action: int = agent.select_action(state, epsilon)
            coords = agent.action_to_coords(action)

            # ── Environment step ─────────────────────────────────────────
            next_state_dict, reward, done = env.step(coords)
            next_state = D3QNAgent.flatten_state(next_state_dict)

            # ── Store transition ─────────────────────────────────────────
            agent.replay_buffer.push(state, action, reward, next_state, done)

            # ── Train step (only once buffer is sufficiently populated) ──
            if len(agent.replay_buffer) >= max(args.min_buffer, args.batch_size):
                loss = agent.train_step(args.batch_size, args.gamma)
                agent.soft_update(args.tau)
                episode_loss += loss
                loss_steps += 1

            episode_reward += reward
            state = next_state

        # ── Epsilon decay ─────────────────────────────────────────────────
        epsilon = max(args.epsilon_end, epsilon * args.epsilon_decay)

        # ── Per-episode logging ──────────────────────────────────────────
        avg_loss_str = (
            f"{episode_loss / loss_steps:10.4f}" if loss_steps > 0 else "          —"
        )
        print(
            f"{episode:>8d} "
            f"{episode_reward:>10.2f} "
            f"{next_state_dict['sum_rate_mbps']:>14.2f} "
            f"{next_state_dict['num_nodes']:>6d} "
            f"{epsilon:>7.4f} "
            f"{avg_loss_str}"
        )

        # ── Periodic checkpoint ───────────────────────────────────────────
        if episode % args.checkpoint_every == 0:
            save_checkpoint(agent, checkpoint_dir, episode)

    # ── Final checkpoint ─────────────────────────────────────────────────
    print("─" * 70)
    save_checkpoint(agent, checkpoint_dir, episode)
    print(f"\nTraining complete. Final weights → {checkpoint_dir / 'd3qn_policy.pth'}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train(parse_args())
