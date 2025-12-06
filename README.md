# Quadcopter Corridor Navigation with PPO and Dreamer (Isaac Lab)

This repository extends **PyDreamerV1-DRL** with a complete experiment suite for
**quadcopter corridor navigation**, implemented in **NVIDIA Isaac Lab (Isaac Sim 5.1)**.
It provides a clean comparison between:

- **PPO (model-free reinforcement learning)**  
- **Dreamer (model-based world-model RL)**  

under two conditions:

1. **No obstacle** (straight corridor)  
2. **Obstacle avoidance** (spherical obstruction at x=6.0 m)

The project evaluates **sample efficiency**, **training stability**, and **final success rate** across both RL paradigms.

---

# 📁 Repository Structure

isaaclab_quadcopter/
│
├── env/
│ └── quadcopter_corridor_env.py # Isaac Lab environment (dynamics, reward, reset)
│
├── ppo/
│ └── train_corridor_ppo.py # PPO training script
│
├── dreamer/
│ └── train_simple_dreamer.py # Simple Dreamer training (world model + actor)
│
├── eval/
│ ├── compare_training.py # Generates PPO vs Dreamer comparison plots
│ └── watch_drone.py # Visualize a single trained drone in Isaac Lab
│
├── raw_logs/
│ ├── ppo_no_obstacle.json
│ ├── ppo_obstacle.json
│ ├── dreamer_no_obstacle.json
│ └── dreamer_obstacle.json # Raw training logs for reproducibility
│
├── results/
│ ├── comparison_no_obstacle.png
│ ├── comparison_obstacle.png # PPO vs Dreamer success-rate plots
│ ├── dreamer_diagnostics_no_obstacle.png
│ └── dreamer_diagnostics_obstacle.png # Dreamer world-model diagnostics
│
└── README.md



---

# 🚁 Task Description

A quadcopter must fly through a **10-meter corridor** to reach a goal region at  
**x ≥ 9.3 meters** while maintaining safe altitude and avoiding walls.

Two experimental conditions:

### **1) No Obstacle**
A straight corridor with no obstruction.

### **2) Obstacle Avoidance**
A rectangular  obstacle centered at x = 6m



Robust collision handling is implemented for:
- wall collisions  
- obstacle collisions  
- ground/ceiling contact  

---

# 🧠 Algorithms Compared

| Component | PPO (Baseline) | Dreamer (Model-Based) |
|----------|----------------|------------------------|
| Type | Model-free, on-policy | Model-based (world model + imagination) |
| Networks | Actor-Critic MLP | Encoder → World Model → Actor & Critic |
| Learning | Directly from environment | Mostly from imagined rollouts |
| Sample Efficiency | Lower | Higher |
| Stability | Can collapse late-training | Stable due to latent dynamics |

The **Dreamer agent** learns a dynamics model for the quadcopter, enabling it to train the policy on *imagined trajectories*, dramatically improving efficiency.

---

# 📊 Experiments & Results

We ran each algorithm for **400k–500k steps** on:

- NVIDIA RTX 5080
- Ubuntu 22.04
- Isaac Lab 5.1
- 32 parallel environments

Evaluation was performed every 10,000 steps.

---

## **Results: No Obstacle**

| Metric | PPO | Dreamer |
|--------|-----|----------|
| **Best Success Rate** | 100.0% | 100.0% |
| **Final Success Rate** | 0.0% | 100.0% |
| **Steps to 50% Success** | 307,200 | 195,008 |
| **Best Avg Distance** | 9.29 m | 9.29 m |

📌 **Interpretation**  
Both methods solved the task, but PPO suffered from **late-training collapse**, a known instability in on-policy RL.  
Dreamer maintained stable success throughout because its world model provides more consistent policy gradients.

<p align="center">
  <img src="../results/comparison_no_obstacle.png" width="600">
</p>

---

## **Results: Obstacle Avoidance**

| Metric | PPO | Dreamer |
|--------|-----|----------|
| **Best Success Rate** | 66.7% | 100.0% |
| **Final Success Rate** | 0.0% | 100.0% |
| **Steps to 50% Success** | 393,216 | 250,016 |
| **Steps to 80% Success** | Not reached | 340,000 |
| **Best Avg Distance** | 8.98 m | 9.29 m |

📌 **Interpretation**  
Obstacle avoidance requires planning — PPO never converged past ~67%.  
Dreamer, using its latent dynamics model, consistently learned to route around the obstacle and reached **100% success**.

<p align="center">
  <img src="../results/comparison_obstacle.png" width="600">
</p>

---

# 🔍 Why Dreamer Outperformed PPO

### **1️⃣ Dreamer sees further into the future**
Imagination rollouts allow 10–30 step lookahead, enabling planning around obstacles.

### **2️⃣ PPO is prone to catastrophic forgetting**
On-policy updates repeatedly overwrite good behaviors.

### **3️⃣ Dreamer's value estimates are smoother**
Because they are computed in latent space, not noisy environment space.

### **4️⃣ Sample efficiency**
Dreamer needs **1.6× fewer steps** to reach 50% success in both tasks.

---

# ▶️ Running the Experiments

### **Train PPO**
```bash
./isaaclab.sh -p isaaclab_quadcopter/ppo/train_corridor_ppo.py --num_envs 16
```
Train Dreamer
```bash
./isaaclab.sh -p isaaclab_quadcopter/dreamer/train_simple_dreamer.py --num_envs 16
```

Compare PPO vs Dreamer
```bash
./_isaac_sim/python.sh isaaclab_quadcopter/eval/compare_training.py \
    --ppo raw_logs/ppo_obstacle.json \
    --dreamer raw_logs/dreamer_obstacle.json \
    --output results/comparison_obstacle.png
```
Watch a Drone Fly
```bash
./isaaclab.sh -p isaaclab_quadcopter/eval/watch_drone.py \
    --checkpoint checkpoints_dreamer/best.pt \
    --obstacle
```

🧪 Reproducibility

Raw logs for all experiments are stored in:
```bash
isaaclab_quadcopter/raw_logs/
```

This includes:
success rates
crash breakdown
average distance
training curves
diagnostics
These logs can be re-evaluated or plotted independently.

📘 Citation

If you use this work, please cite:

@software{IsaacLabQuadcopter2025,
  author = {Philip, Sandeep John},
  title = {Quadcopter Corridor Navigation: PPO vs Dreamer in Isaac Lab},
  year = {2025},
  url = {https://github.com/razarcon3/PyDreamerV1-DRL}
}

🙌 Acknowledgements

NVIDIA Isaac Lab team
PyDreamer original authors (Hafner et al.)
Georgia Tech DRL course (CS 8803 DRL)
Collaborators & reviewers

