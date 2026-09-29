<!-- 
==============================================================================
Author: Kamil BENMADI
Email: kamil.benmadi@sigma-clermont.fr
GitHub: https://github.com/Uncrowned0x0
==============================================================================
-->

# 🎓 Beginner's Step-by-Step Guide — KAIROS Reinforcement Learning (PPO & Curriculum)

> **Who is this guide for?**  
> This guide is written for anyone starting with **Reinforcement Learning (RL)**, **ROS 2**, and **Docker**. All commands are provided as **raw, standard terminal commands** (no wrapper scripts required), directly suitable for GitHub reproducibility and automated pipelines.

---

## 📋 Table of Contents

- [Step 0 — Overview & Curriculum Learning](#step-0--overview--curriculum-learning)
- [Step 1 — Verify Host Prerequisites](#step-1--verify-host-prerequisites)
- [Step 2 — Authorize 3D GUI Display (X11)](#step-2--authorize-3d-gui-display-x11)
- [Step 3 — Build the Docker Image (PyTorch + CUDA)](#step-3--build-the-docker-image-pytorch--cuda)
- [Step 4 — Start the RL Container](#step-4--start-the-rl-container)
- [Step 5 — Compile the ROS 2 Workspace (Colcon Build)](#step-5--compile-the-ros-2-workspace-colcon-build)
- [Step 6 — Run Unit Tests](#step-6--run-unit-tests)
- [Step 7 — First Demo: Evaluate a Pretrained Model](#step-7--first-demo-evaluate-a-pretrained-model)
- [Step 8 — Train a Model with Curriculum Learning](#step-8--train-a-model-with-curriculum-learning)
- [Step 9 — Track Learning Progress with TensorBoard](#step-9--track-learning-progress-with-tensorboard)
- [Step 10 — Interactive Keyboard Teleoperation](#step-10--interactive-keyboard-teleoperation)
- [Step 11 — Clean Shutdown & Container Teardown](#step-11--clean-shutdown--container-teardown)
- [❓ FAQ & Troubleshooting](#-faq--troubleshooting)

---

## Step 0 — Overview & Curriculum Learning

### How Does RL Work on KAIROS?
Unlike classical trajectory planning where an engineer writes analytical equations, in **Reinforcement Learning**, the robot (**Agent**) learns by trial and error using the **PPO (Proximal Policy Optimization)** algorithm:
1. **Observation**: UR5e joint positions and velocities, gripper state, target object 3D coordinates.
2. **Action**: Continuous delta joint positions/velocities applied to the UR5e arm.
3. **Reward**: Gazebo physics evaluates the action:
   - ➕ **Bonus**: Progress towards the target cube or successful grasp and lift.
   - ➖ **Penalty**: Collisions with the table/chassis or abrupt joint motions.

### 4-Stage Curriculum Learning System
- **Level 0 (Reach)**: Arm reaches any random Cartesian coordinate in 3D workspace.
- **Level 1 (Pick)**: Approaches a cube on a table, closes gripper, and lifts it.
- **Level 2 (Place)**: Starts with cube in gripper, transports and releases it inside a bin.
- **Level 3 (Full)**: Complete autonomous Pick & Place sequence from scratch.

---

## Step 1 — Verify Host Prerequisites

Open a terminal on your host machine (`Ctrl+Alt+T`) and run:

```bash
# 1. Check Docker installation
docker --version
# Expected: Docker version 24.x.x or newer

# 2. Check Docker Compose
docker compose version

# 3. Check NVIDIA GPU driver
nvidia-smi
# Expected: Your GPU model (e.g. GeForce RTX 4070) with driver status
```

---

## Step 2 — Authorize 3D GUI Display (X11)

Allow Docker containers to display Gazebo and RViz windows on your desktop:

```bash
# Authorize local container connections to the X11 server
xhost +local:docker
```

---

## Step 3 — Build the Docker Image (PyTorch + CUDA)

Navigate to the RL workspace on your host:

```bash
cd /home/YourMachine/kairos_ws_rl
```

Build the Docker image with NVIDIA GPU support:

```bash
# With NVIDIA GPU acceleration:
LOCAL_UID=$(id -u) LOCAL_GID=$(id -g) docker compose -f docker-compose.yaml -f docker-compose.gpu.yaml build

# Or CPU-only mode:
LOCAL_UID=$(id -u) LOCAL_GID=$(id -g) docker compose -f docker-compose.yaml build
```

> ⏳ *Initial build takes 5 to 15 minutes to download PyTorch CUDA runtime, Gazebo Harmonic, and dependencies. Subsequent builds use Docker layer cache and take seconds.*

---

## Step 4 — Start the RL Container

Start the container in daemon mode:

```bash
# With GPU:
LOCAL_UID=$(id -u) LOCAL_GID=$(id -g) docker compose -f docker-compose.yaml -f docker-compose.gpu.yaml up -d

# Or CPU-only:
LOCAL_UID=$(id -u) LOCAL_GID=$(id -g) docker compose -f docker-compose.yaml up -d
```

Verify that `kairos_rl_sim` is active:

```bash
docker ps --filter "name=kairos_rl_sim"
```

---

## Step 5 — Compile the ROS 2 Workspace (Colcon Build)

Open an interactive bash terminal inside the container:

```bash
docker exec -it -u robot kairos_rl_sim bash
```

Inside the container shell (`robot@...:~/ros2_ws$`):

```bash
cd /home/robot/ros2_ws
source /opt/ros/jazzy/setup.bash

# Compile packages with Release optimizations
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

# Source install overlay
source install/setup.bash
```

> 💡 *All 31 packages should compile with zero errors.*

---

## Step 6 — Run Unit Tests

Verify kinematics, collision detection, and curriculum environments:

```bash
# Inside the container:
source /opt/ros/jazzy/setup.bash && source install/setup.bash
python3 -m pytest -v src/kairos_rl/test/test_kinematics_and_env.py

# Or in 1 line directly from host terminal:
docker exec -it -u robot kairos_rl_sim bash -c "source /opt/ros/jazzy/setup.bash && source install/setup.bash && python3 -m pytest -v src/kairos_rl/test/test_kinematics_and_env.py"
```

Expected output:
```text
============================== 8 passed in ~2.5s ===============================
```

---

---

## Step 7 — Modular Multi-Terminal Workflow (Recommended)

To achieve maximum modularity, stability, and clean debugging, the simulation is split across separate terminals:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       MODULAR MULTI-TERMINAL ARCHITECTURE                   │
│                                                                             │
│   Terminal 1: Gazebo World Server & GUI (/clock + physics)                  │
│       │                                                                     │
│   Terminal 2: RB-KAIROS Robot Spawn + Controllers + RViz2                   │
│       │                                                                     │
│   Terminal 3: RL Training / Policy Evaluation OR Keyboard Teleoperation     │
│       │                                                                     │
│   Terminal 4: TensorBoard Live Training Telemetry                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

> ⚡ **NVIDIA GPU Hardware Acceleration:**  
> `docker-compose.yaml` has NVIDIA GPU hardware acceleration reservations enabled by default (`deploy.resources.reservations.devices` and `runtime: nvidia`). Gazebo Harmonic uses Ogre 2.x on your RTX 4070 GPU rather than falling back to CPU software rasterization (`llvmpipe`), ensuring smooth 60 FPS real-time rendering.

### 🖥️ Terminal 1 — Start Gazebo Simulation World

In your first terminal on the host, launch the Gazebo simulation world alone (default: `labo`):

**Raw Terminal Command:**
```bash
docker exec -it -u robot kairos_rl_sim bash -c "source /opt/ros/jazzy/setup.bash && source /home/robot/ros2_ws/install/setup.bash && ros2 launch kairos_bringup kairos_world.launch.py world:=labo"
```

*Or via convenience wrapper:*
```bash
./kairos_rl.sh world labo
```

---

### 🤖 Terminal 2 — Spawn Robot, Controllers & RViz2

Open a **second terminal** on your host machine to spawn the RB-KAIROS robot and its controllers.

> ✋ **Strict Gripper Filtering:**  
> The simulation supports **only two end-effectors**:
> 1. `schunk_egk50` — Schunk 2-finger parallel gripper (default)
> 2. `tesollo_dg5f` — Tesollo DG-5F-R 5-finger dexterous hand with 20 DOFs  
> *(Obsolete grippers `qbhand` and `onrobot_rg6` are completely disabled).*

**Raw Terminal Command (with Schunk EGK50 Gripper):**
```bash
docker exec -it -u robot kairos_rl_sim bash -c "source /opt/ros/jazzy/setup.bash && source /home/robot/ros2_ws/install/setup.bash && ros2 launch kairos_bringup kairos_robot.launch.py gripper_type:=schunk_egk50"
```

**Raw Terminal Command (with Tesollo DG-5F-R Hand):**
```bash
docker exec -it -u robot kairos_rl_sim bash -c "source /opt/ros/jazzy/setup.bash && source /home/robot/ros2_ws/install/setup.bash && ros2 launch kairos_bringup kairos_robot.launch.py gripper_type:=tesollo_dg5f"
```

*Or via convenience wrapper:*
```bash
./kairos_rl.sh robot schunk_egk50
# or
./kairos_rl.sh robot tesollo_dg5f
```

*(Optional: For standalone RViz2 in a separate terminal: `ros2 launch kairos_bringup kairos_rviz.launch.py` or `./kairos_rl.sh rviz`).*

---

### 🧠 Option A — Reinforcement Learning (Train / Eval - Single Command)

> 💡 **IMPORTANT:** For Reinforcement Learning, **a single command is enough**! The scripts (`train_curriculum` and `eval_curriculum`) launch Gazebo, the robot, and RViz automatically.
> Launching the world and the robot manually in Terminal 1 and 2 is only used to verify that everything is fully functional, or to do manual teleoperation.

You only need **one terminal** on your host machine for RL experiments:

#### Option 1: Evaluate Pretrained Model
Load and visually run the pretrained **Level 0 (Reach)** neural network checkpoint:

**Raw Terminal Command:**
```bash
docker exec -it -u robot kairos_rl_sim bash -c "source /opt/ros/jazzy/setup.bash && source /home/robot/ros2_ws/install/setup.bash && ros2 run kairos_rl eval_curriculum --level 0 --checkpoint ./checkpoints/level0_final.zip --episodes 5"
```

*Or via convenience wrapper:*
```bash
./kairos_rl.sh eval --level 0
```

#### Option 2: Train with Curriculum Learning
Train Level 0 (Reach) or continue transfer learning on Level 1 (Pick):

**Raw Terminal Command:**
```bash
docker exec -it -u robot kairos_rl_sim bash -c "source /opt/ros/jazzy/setup.bash && source /home/robot/ros2_ws/install/setup.bash && ros2 run kairos_rl train_curriculum --level 0 --num-envs 2 --total-timesteps 500000"
```

*Or via convenience wrapper:*
```bash
./kairos_rl.sh train --level 0 --num-envs 2
```

### 🎮 Option B — Interactive Teleoperation (Manual Verification)

If you launched Gazebo and the robot manually (Terminal 1 and 2), you can open a **third terminal** to drive the robot:

**Raw Terminal Command:**
```bash
docker exec -it -u robot kairos_rl_sim bash -c "source /opt/ros/jazzy/setup.bash && source /home/robot/ros2_ws/install/setup.bash && ros2 launch kairos_bringup kairos_teleop.launch.py"
```

*Or via convenience wrapper:*
```bash
./kairos_rl.sh teleop
```

---


## Step 8 — Train a Model with Curriculum Learning (Details)

### Train Level 0 (Reach) from Scratch:
```bash
docker exec -it -u robot kairos_rl_sim bash -c "source /opt/ros/jazzy/setup.bash && source install/setup.bash && ros2 run kairos_rl train_curriculum --level 0 --num-envs 4 --total-timesteps 500000"
```

### Train Level 1 (Pick — Grasp Cube) using Transfer Learning:
Resume from Level 0 pretrained weights:
```bash
docker exec -it -u robot kairos_rl_sim bash -c "source /opt/ros/jazzy/setup.bash && source install/setup.bash && ros2 run kairos_rl train_curriculum --level 1 --resume ./checkpoints/level0_final.zip --num-envs 4 --total-timesteps 1000000"
```

> 💾 **Automatic Checkpoints:** Models are saved directly to your host's `./checkpoints/` directory every 100,000 steps.

---

## Step 9 — Track Learning Progress with TensorBoard

Open a new terminal on your host to start the TensorBoard web dashboard:

```bash
docker exec -it -u robot kairos_rl_sim tensorboard --logdir=/home/robot/ros2_ws/tb_logs --port=6006 --bind_all
```

Open your browser and navigate to: **[http://localhost:6006](http://localhost:6006)**  
You can monitor live training curves: `rollout/ep_rew_mean`, `train/policy_gradient_loss`, `train/value_loss`, and success rates.

---

## Step 10 — Interactive Keyboard Teleoperation

To manually move the robot base using the arrow keys and test actuators:

```bash
docker exec -it -u robot kairos_rl_sim bash -c "source /opt/ros/jazzy/setup.bash && source install/setup.bash && python3 /home/robot/ros2_ws/kairos_teleop_keyboard.py"
```

### Controls Reference:
- **▲ Up Arrow** / **▼ Down Arrow** : Forward / Backward
- **◄ Left Arrow** / **► Right Arrow** : Turn Left / Turn Right
- **Shift + ◄ / ►** : Lateral Crabbing (Mecanum)
- **`+`** / **`-`** : Increase / Decrease Speed
- **Spacebar** : Emergency Stop ($0\text{ m/s}$)

---

## Step 11 — Clean Shutdown & Container Teardown

1. Stop training by pressing `Ctrl+C` in the terminal (gracefully saves checkpoint before exit).
2. Stop the Docker container on the host:

```bash
cd /home/YourMachine/kairos_ws_rl
docker compose down
```

---

## ❓ FAQ & Troubleshooting

### Q1: Where are trained models saved?
Checkpoints are saved on your host machine in `/home/YourMachine/kairos_ws_rl/checkpoints/` (bind-mounted directly from the container).

### Q2: How to clean up shared memory after an interrupted training session?
```bash
docker exec -u root kairos_rl_sim rm -rf /dev/shm/* /home/robot/.ros/log/*
```

---

*Author: Kamil BENMADI (<kamil.benmadi@sigma-clermont.fr>) — [GitHub](https://github.com/Uncrowned0x0) | [LinkedIn](https://www.linkedin.com/in/kamilb-)*
