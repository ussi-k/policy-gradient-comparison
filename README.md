## 🧠 Policy Gradient Variants Comparison

### 📌 Overview

This project implements and compares three variants of the Policy Gradient (REINFORCE) algorithm in Reinforcement Learning.

The goal is to understand how different techniques affect learning speed, stability, and performance.

---

### ⚙️ Environment & Tools

* Environment: CartPole-v1
* Framework: PyTorch
* Library: Gymnasium
* Language: Python

---

### 🚀 Implemented Methods

#### 1. Basic Policy Gradient

* Uses total episode reward for all time steps
* Simple implementation
* Suffers from high variance

---

#### 2. Reward-to-Go

* Assigns each step the sum of future rewards only
* Improves credit assignment
* Leads to faster and more stable learning

---

#### 3. Baseline (Advantage Function)

* Introduces a value function as a baseline
* Uses: Advantage = Reward-to-Go − Value(s)
* Reduces variance in updates
* Provides more stable training

---

### 📊 Results

![Comparison](https://chatgpt.com/c/results/comparison.png)

---

### 🔍 Key Insights

* Basic Policy Gradient is noisy due to uniform reward assignment
* Reward-to-Go improves learning speed
* Baseline reduces variance and stabilizes training
* Combining Reward-to-Go with a baseline gives the best performance

---

### 📁 Project Structure

policy-gradient-project/
│
├── basic_pg.py
├── rtg_pg.py
├── baseline_pg.py
│
├── plot_results.py
│
├── results/
│ ├── basic.npy
│ ├── rtg.npy
│ ├── baseline.npy
│ ├── comparison.png
│
├── README.md
├── requirements.txt

---

### ▶️ How to Run

pip install -r requirements.txt

python basic_pg.py
python rtg_pg.py
python baseline_pg.py
python plot_results.py

---

### 🧠 What I Learned

* How policy gradient methods maximize expected reward
* The importance of reducing variance in reinforcement learning
* How Reward-to-Go improves credit assignment
* How baselines stabilize training using value functions

---

### 🔗 Future Improvements

* Implement Actor-Critic methods
* Test on more complex environments
* Add hyperparameter tuning
* Compare with PPO or A2C
