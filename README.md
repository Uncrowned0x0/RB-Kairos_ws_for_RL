<!-- 
==============================================================================
Author: Kamil BENMADI
Email: kamil.benmadi@sigma-clermont.fr
GitHub: https://github.com/Uncrowned0x0
==============================================================================
-->

# Kairos WS RL 🧠🤖

Workspace dedicated to **Reinforcement Learning (RL)** for the **Robotnik RB-KAIROS+** mobile manipulator equipped with a **UR5e** arm and a modular gripper (**Schunk EGK50** or **Tesollo DG-5F-R**).

Integrates a high-performance simulation environment under **ROS 2 Jazzy**, **Gazebo Harmonic**, **PyTorch**, and **Stable-Baselines3 (PPO)** with a complete **4-level Curriculum Learning system**.

---

## Table of Contents

1. [Overview & Philosophy](#overview--philosophy)
2. [Workspace Architecture](#workspace-architecture)
3. [4-Level Curriculum Learning System](#4-level-curriculum-learning-system)
4. [Convenience Script `kairos_rl.sh`](#convenience-script-kairos_rlsh)
5. [Docker Configuration & GPU Acceleration](#docker-configuration--gpu-acceleration)
6. [Quickstart Guide](#quickstart-guide)
7. [Checkpoint Evaluation & Visual Demonstration](#checkpoint-evaluation--visual-demonstration)
8. [Model Training](#model-training)
9. [TensorBoard Monitoring](#tensorboard-monitoring)
10. [Included Packages & Applied Fixes](#included-packages--applied-fixes)
11. [Troubleshooting & FAQ](#troubleshooting--faq)

---

## Overview & Philosophy

Unlike a general simulation workspace (such as `kairos_ws_light`), `kairos_ws_rl` is **specifically optimized for Reinforcement Learning**:

- **Strict Isolation:** Only packages strictly necessary for the UR5e arm, Schunk / Tesollo grippers, sensors, and the RL Gym interface are retained.
- **RL-Optimized URDF:** Uses `rbkairos_ur5_rl.urdf.xacro` which anchors the base to the ground (`world_to_base`) to accelerate arm and grasping training without mobile base instability or unnecessary LiDAR raycasting.
- **Multi-Instance Parallelism:** Executes up to 8 to 12 concurrent Gazebo simulations via `SubprocVecEnv` (using partitioned `ROS_DOMAIN_ID` and `GZ_PARTITION`). Only worker 0 renders the 3D GUI window; others run headless for maximum FPS.
- **Robustness & Transfer Learning:** Partial weight-loading mechanism across curriculum level transitions when observation and action spaces expand.

---

## Workspace Architecture

```
kairos_ws_rl/
├── Dockerfile                   ← ROS 2 Jazzy + Gazebo Harmonic + PyTorch + SB3 image
├── docker-compose.yaml          ← Docker Compose with NVIDIA GPU acceleration by default
├── docker-compose.gpu.yaml      ← Dedicated NVIDIA GPU override configuration
├── kairos_rl.sh                 ← ✨ Management script (world, robot, rviz, train, eval, sim...)
├── README.md                    ← Technical documentation
├── instructions_rl.md             ← Step-by-step instructions for RL (Terminal/Docker details)
├── checkpoints/                 ← Active checkpoints and milestones (level0_final.zip, etc.)
│   ├── level0_final.zip
│   ├── kairos_level0_reach_1000000_steps.zip
│   ├── kairos_level0_reach_2000000_steps.zip
│   ├── kairos_level0_reach_3000000_steps.zip
│   └── kairos_level0_reach_4000000_steps.zip
├── rl_level0/                   ← Archive of 161 training checkpoints (0 to 4M steps)
├── tb_logs/                     ← Persistent TensorBoard run logs
├── docker/
│   ├── kairos-rl-entrypoint.sh  ← Container entrypoint with FastRTPS shared memory cleanup
│   └── requirements/
│       ├── builder/packages.txt ← Build essentials, X11, gosu, procps
│       ├── base/packages.txt    ← ROS 2 Jazzy, Gazebo Harmonic, controllers
│       └── python/requirements.txt ← PyTorch, stable-baselines3, gymnasium, tensorboard
└── src/
    ├── kairos_rl/               ← ✨ Core RL package (Gym environments, trainers, eval)
    │   ├── kairos_rl/
    │   │   ├── curriculum/      ← Curriculum levels 0, 1, 2, 3
    │   │   │   ├── level0_reach.py
    │   │   │   ├── level1_pick.py
    │   │   │   ├── level2_place.py
    │   │   │   └── level3_full.py
    │   │   ├── eval_curriculum.py ← ✨ Evaluation script with 3D visualization
    │   │   ├── train_curriculum.py← Multi-instance curriculum runner
    │   │   ├── train_ppo.py     ← Single-instance trainer
    │   │   ├── train_ppo_parallel.py ← Parallel trainer
    │   │   ├── gz_utils.py      ← Dynamic Gazebo object spawning utilities
    │   │   └── ros_interface.py ← ROS 2 Reliable QoS bridge with ros2_control
    │   ├── config/              ← curriculum_params.yaml and rl_params.yaml
    │   ├── launch/              ← train.launch.py
    │   └── test/                ← Unit test suite (test_kinematics_and_env.py)
    ├── rbkairos_description/    ← Robot descriptions, RL xacro (rbkairos_ur5_rl.urdf.xacro)
    ├── robotnik/                ← Gazebo Harmonic simulation, labo.world, spawn nodes
    ├── schunk_egk50_description/← Schunk EGK50 electric gripper model and controller
    ├── delto_m_ros2/            ← Tesollo DG-5F-R gripper model and plugins
    ├── tesollo_tactile_mock/    ← Mock simulation of tactile taxels
    ├── xela_description/        ← Xela sensor geometric description
    ├── qb_hand_description/     ← Obsolete gripper model (disabled via COLCON_IGNORE)
    ├── Universal_Robots_ROS2_Description/ ← UR5e arm description (marked COLCON_IGNORE)
    └── kairos_bringup/          ← Modular multi-terminal bringup launch files:
        ├── kairos_world.launch.py         ← Terminal 1: Gazebo world alone (labo)
        ├── kairos_robot.launch.py         ← Terminal 2: Robot spawn + controllers + RViz2
        ├── kairos_rviz.launch.py          ← Standalone RViz2
        ├── kairos_teleop.launch.py        ← Terminal 3: Keyboard teleoperation
        └── kairos_sim_complete.launch.py  ← Full simulation shortcut
```

---

## 4-Level Curriculum Learning System

Complex mobile manipulation (Pick & Place with 6 DoF arm + gripper) converges slowly when trained end-to-end directly. The curriculum breaks down the learning process into 4 progressive stages:

| Level | Name | Description | Actions | Observations |
|---|---|---|---|---|
| **Level 0** | **Reach** | Arm learns to position its TCP at random 3D XYZ target points above the table. Fixed base, gripper closed. | $\Delta q_1..\Delta q_6$ (6D) | Joint pos, vel, TCP, TCP→target, distance (16D) |
| **Level 1** | **Pick** | Arm + gripper learn to approach a physical cube, grasp it, and lift it above the table. Uses Level 0 weights. | $\Delta q_1..\Delta q_6$ + gripper (7D) | Joint pos, vel, grip, TCP→cube, grasp status (17D) |
| **Level 2** | **Place** | Starts with cube in gripper, learns to transport it safely into a destination bin without collisions or premature release. | $\Delta q_1..\Delta q_6$ + gripper (7D) | Joint pos, vel, grip, cube→bin, grasp status (17D) |
| **Level 3** | **Full** | Complete end-to-end task: rest $\rightarrow$ reach $\rightarrow$ pick $\rightarrow$ transport $\rightarrow$ place. | $\Delta q_1..\Delta q_6$ + gripper (7D) | Joint pos, vel, grip, TCP→cube, cube→bin, phase (21D) |

### Built-in Safety Features
- **Analytical Collision Checking:** Geometric boundary and collision checks against KAIROS chassis, table undersurface, and self-collision at every timestep. Penalty: `-200.0`.
- **Safe Reset:** If an episode terminates with the arm too low, `reset()` executes an upward shoulder lift before returning to the HOME pose to prevent hooking the table.
- **Frontal Workspace Sampling:** Targets and cubes are sampled in the frontal workspace ($[-1.2, 1.2]$ rad), avoiding unreachable configurations behind the base.

---

## Convenience Script `kairos_rl.sh`

The `./kairos_rl.sh` script centralizes all operations, automatically configuring NVIDIA GPU acceleration when available.

```bash
# Docker lifecycle
./kairos_rl.sh build                 # Build Docker image (Jazzy + Gazebo + PyTorch)
./kairos_rl.sh start                 # Start container in background
./kairos_rl.sh stop                  # Stop container
./kairos_rl.sh restart               # Restart container
./kairos_rl.sh shell                 # Open interactive bash shell (user: robot)
./kairos_rl.sh status                # Display status of container, GPU, checkpoints, logs
./kairos_rl.sh clean                 # Purge shared memory (/dev/shm) and zombie processes

# Build & tests
./kairos_rl.sh colcon_build          # Compile ROS 2 packages
./kairos_rl.sh test                  # Run unit tests (kinematics, collisions, config)

# Modular Multi-Terminal Simulation Workflow (Recommended)
./kairos_rl.sh world [world]             # Terminal 1: Launch Gazebo world alone (default: labo)
./kairos_rl.sh robot [gripper] [rviz]    # Terminal 2: Spawn RB-KAIROS robot + RViz2
                                         #   gripper: schunk_egk50 (default) | tesollo_dg5f
                                         #   rviz: true (default) | false
./kairos_rl.sh rviz                      # Terminal 2/4: Launch standalone RViz2
./kairos_rl.sh sim [world] [gripper]     # Combined shortcut (Gazebo + Robot + RViz)

# Evaluation & 3D visualization (Terminal 3)
./kairos_rl.sh eval                      # Evaluate Level 0 model (5 episodes in Gazebo)
./kairos_rl.sh eval --level 0 --checkpoint ./checkpoints/level0_final.zip --episodes 10

# Reinforcement Learning training (Terminal 3)
./kairos_rl.sh train --level 0 --num-envs 4 --total-timesteps 500000
./kairos_rl.sh train --level 1 --resume ./checkpoints/level0_final.zip --num-envs 8
./kairos_rl.sh train_ppo                 # Standard single-instance PPO
./kairos_rl.sh train_parallel --num-envs 2

# Monitoring
./kairos_rl.sh tensorboard               # Start TensorBoard at http://localhost:6006

# Manual teleoperation (Terminal 3)
./kairos_rl.sh teleop                    # Keyboard teleoperation (arrows, crabbing, speed)
./kairos_rl.sh teleop xterm              # In dedicated xterm window
```

---

## Docker Configuration & GPU Acceleration

The container is built from `osrf/ros:jazzy-desktop` with Gazebo Harmonic and PyTorch preinstalled.

### NVIDIA GPU Acceleration
When an NVIDIA GPU (e.g. GeForce RTX 4070) is detected, `./kairos_rl.sh` automatically overlays `docker-compose.gpu.yaml`. This enables:
- Hardware OpenGL rendering for Gazebo (`libgl1-mesa-dri`, `nvidia-container-toolkit`).
- CUDA computation for PyTorch.

To force CPU mode:
```bash
FORCE_CPU=1 ./kairos_rl.sh start
```

---

## Quickstart Guide

1. **Authorize X11 display:**
   ```bash
   xhost +local:docker
   ```
2. **Start container:**
   ```bash
   cd /home/YourMachine/kairos_ws_rl
   ./kairos_rl.sh start
   ```
3. **Compile ROS 2 packages:**
   ```bash
   ./kairos_rl.sh colcon_build
   ```
4. **Run unit tests:**
   ```bash
   ./kairos_rl.sh test
   ```
5. **Evaluate pretrained Level 0 checkpoint:**
   ```bash
   ./kairos_rl.sh eval --level 0 --checkpoint ./checkpoints/level0_final.zip --episodes 5
   ```

---

## Modular Multi-Terminal Simulation Workflow (Recommended)

Run simulation components in separate terminals for independent lifecycle control and clean debugging:

```bash
# Terminal 1 — Start Gazebo Simulation World (default: labo)
./kairos_rl.sh world labo

# Terminal 2 — Spawn RB-KAIROS Robot, Controllers & RViz2
# Supported grippers: schunk_egk50 (default) | tesollo_dg5f
./kairos_rl.sh robot schunk_egk50

# (Optional: launch RViz2 separately if passing rviz:=false to robot command)
./kairos_rl.sh rviz

# Terminal 3 — RL Training / Evaluation OR Teleoperation
./kairos_rl.sh eval --level 0                     # Evaluate pretrained policy
# Or:
./kairos_rl.sh train --level 0 --num-envs 2        # Train curriculum level 0
# Or:
./kairos_rl.sh teleop                              # Keyboard teleoperation
```

---

## Checkpoint Evaluation & Visual Demonstration

The `eval_curriculum` script evaluates any `.zip` checkpoint inside Gazebo:

```bash
./kairos_rl.sh eval --level 0 --checkpoint ./checkpoints/level0_final.zip --episodes 5
```

Evaluation options:
- `--level <0|1|2|3>`: Environment level.
- `--checkpoint <path>`: Checkpoint `.zip` file path.
- `--episodes <N>`: Number of evaluation episodes.
- `--gui` / `--no-gui`: Enable/disable Gazebo 3D GUI.
- `--no-spawn-sim`: Attach to an already running simulation.
- `--stochastic`: Sample actions stochastically instead of deterministic mean.

---

## Model Training

### Multi-Instance Curriculum Learning (Recommended)

```bash
# Level 0: Reach (500k timesteps)
./kairos_rl.sh train --level 0 --num-envs 4 --total-timesteps 500000

# Level 1: Pick (Transfer Learning from Level 0)
./kairos_rl.sh train --level 1 --resume ./checkpoints/level0_final.zip --num-envs 8 --total-timesteps 1000000

# Level 2: Place (Transfer Learning from Level 1)
./kairos_rl.sh train --level 2 --resume ./checkpoints/kairos_level1_pick_1000000_steps.zip --num-envs 8

# Level 3: Full End-to-End
./kairos_rl.sh train --level 3 --resume ./checkpoints/level2_final.zip --num-envs 8
```

Checkpoints are automatically saved directly into host `./checkpoints/`.

---

## TensorBoard Monitoring

1. Launch TensorBoard:
   ```bash
   ./kairos_rl.sh tensorboard
   ```
2. Open browser at: **[http://localhost:6006](http://localhost:6006)**
3. Monitor:
   - `rollout/ep_rew_mean`: Mean episode reward.
   - `rollout/ep_len_mean`: Mean episode length.
   - `train/value_loss` and `train/policy_gradient_loss`: Critic and Actor network convergence.

---

## Included Packages & Applied Fixes

### Included Packages
- `kairos_rl`: Gym environments, PPO trainers, curriculum learning, evaluation tools, unit tests.
- `rbkairos_description`: RB-KAIROS+ models and RL fixed-base xacro (`rbkairos_ur5_rl.urdf.xacro`).
- `robotnik`: Gazebo Harmonic simulation, `labo.world`, spawn scripts, sensor models.
- `schunk_egk50_description`: Schunk EGK50 gripper model and controller.
- `delto_m_ros2`: Tesollo DG-5F-R anthropomorphic hand model and Gazebo plugins.
- `tesollo_tactile_mock` & `xela_description`: Tactile taxel simulation mock.
- `qb_hand_description`: Alternative gripper URDF.
- `kairos_bringup`: Complete simulation, teleoperation, and Nav2 launchers.

### Key Fixes Applied
- **Permission & Docker Startup Fix:** Entrypoint handles root vs non-root seamlessly, safely manages volume permissions without `EPERM` crashes.
- **Docker Build pip Fix:** Added `--ignore-installed` to `pip install` preventing `Cannot uninstall kiwisolver: RECORD file not found` failure on Ubuntu Noble.
- **TTY Handling:** `docker_exec` dynamically detects interactive TTY (`-it` vs `-i`) preventing batch/headless pipeline errors.
- **Dynamic XACRO Resolver:** Python scripts resolve `rbkairos_ur5_rl.urdf.xacro` dynamically via ROS 2 Share and relative workspace fallbacks.
- **Reliable QoS on `/joint_states`:** `ros_interface.py` uses `RELIABLE` QoS to ensure accurate joint feedback from `joint_state_broadcaster`.
- **English Language Standardization:** All terminal outputs, code comments, launch files, Dockerfiles, and documentation (.md files) standardized to English.

---

## Troubleshooting & FAQ

### 1. Gazebo Black Screen or Crash
Run `xhost +local:docker` on the host, and ensure `nvidia-smi` works.

### 2. FastRTPS Shared Memory Errors (`No space left on device`)
```bash
./kairos_rl.sh clean
```

### 3. Checkpoint File Access
Checkpoints are directly located on the host in `/home/YourMachine/kairos_ws_rl/checkpoints/`.

---

*Author: Kamil BENMADI (<kamil.benmadi@sigma-clermont.fr>) — [GitHub](https://github.com/Uncrowned0x0) | [LinkedIn](https://www.linkedin.com/in/kamilb-)*
