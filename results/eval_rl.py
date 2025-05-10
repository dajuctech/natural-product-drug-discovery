import matplotlib.pyplot as plt
import json

print("📊 Plotting RL reward trends...")
with open("outputs/rl_rewards.json") as f:
    rewards = json.load(f)

plt.plot(rewards)
plt.title("RL Optimization Reward Over Time")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.savefig("results/plots/rl_reward_curve.png")
print("✅ RL reward curve saved.")
