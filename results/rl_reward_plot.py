# results/rl_reward_plot.py
import matplotlib.pyplot as plt
import pandas as pd

log_path = "outputs/rl_reward_log.csv"
df = pd.read_csv(log_path)

plt.figure(figsize=(8, 4))
plt.plot(df["episode"], df["reward"], label="Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("RL Reward over Episodes")
plt.legend()
plt.tight_layout()
plt.savefig("results/plots/rl_reward_curve.png")
