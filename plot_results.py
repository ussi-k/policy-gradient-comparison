import numpy as np
import matplotlib.pyplot as plt
import os

# --------- FILE PATHS ---------
basic_path = "results/basic.npy"
rtg_path = "results/rtg.npy"
baseline_path = "results/baseline.npy"

# --------- CHECK FILES EXIST ---------
for path in [basic_path, rtg_path, baseline_path]:
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        exit()

# --------- LOAD DATA ---------
basic = np.load(basic_path)
rtg = np.load(rtg_path)
baseline = np.load(baseline_path)

# --------- OPTIONAL: SMOOTHING FUNCTION ---------
def smooth(data, window=5):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')

# Apply smoothing (optional but looks better)
basic_s = smooth(basic)
rtg_s = smooth(rtg)
baseline_s = smooth(baseline)

# --------- PLOT ---------
plt.figure(figsize=(10, 6))

plt.plot(basic_s, label="Basic Policy Gradient")
plt.plot(rtg_s, label="Reward-to-Go")
plt.plot(baseline_s, label="Baseline (Advantage)")

plt.xlabel("Epoch")
plt.ylabel("Average Return")
plt.title("Policy Gradient Variants Comparison")

plt.legend()
plt.grid(True)

# --------- SAVE FIGURE ---------
save_path = "results/comparison.png"
plt.savefig(save_path)
print(f" Plot saved at: {save_path}")

# --------- SHOW ---------
plt.savefig("results/comparison.png")
plt.show()