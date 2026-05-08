import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.997
WINDOW = 50

df = pd.read_csv("training_log.csv")
episodes = df["episode"].values
rewards = df["reward"].values

epsilon = np.maximum(EPSILON_END, EPSILON_START * (EPSILON_DECAY ** (episodes - 1)))
moving_avg = pd.Series(rewards).rolling(window=WINDOW, min_periods=1).mean().values

fig, ax1 = plt.subplots(figsize=(12, 6))

color_reward = "#2196F3"
ax1.plot(episodes, moving_avg, color=color_reward, linewidth=1.8, label=f"{WINDOW}-ep Moving Avg Reward")
ax1.set_xlabel("Episode", fontsize=13)
ax1.set_ylabel("Reward (50-ep Moving Avg)", color=color_reward, fontsize=13)
ax1.tick_params(axis="y", labelcolor=color_reward)

ax2 = ax1.twinx()
color_eps = "#FF5722"
ax2.plot(episodes, epsilon, color=color_eps, linewidth=1.5, linestyle="--", label="Epsilon")
ax2.set_ylabel("Epsilon", color=color_eps, fontsize=13)
ax2.tick_params(axis="y", labelcolor=color_eps)
ax2.set_ylim(0, 1.05)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=11)

plt.title("DRL Training: Reward & Epsilon Decay vs Episode", fontsize=14)
fig.tight_layout()
plt.savefig("learning_curve.png", dpi=150)
print("Saved learning_curve.png")
