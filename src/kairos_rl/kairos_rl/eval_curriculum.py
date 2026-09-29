# ==============================================================================
# Author: Kamil BENMADI
# Email: kamil.benmadi@sigma-clermont.fr
# GitHub: https://github.com/Uncrowned0x0
# ==============================================================================

"""
Policy Evaluation Script for KAIROS Curriculum Learning.
Enables testing and visual demonstration of trained checkpoints in Gazebo.

Usage:
  # Evaluate Level 0 (Reach) with 10 episodes
  ros2 run kairos_rl eval_curriculum --level 0 --checkpoint ./checkpoints/level0_final.zip --episodes 10

  # Evaluate without respawning Gazebo (if simulation is already running)
  ros2 run kairos_rl eval_curriculum --level 0 --checkpoint ./checkpoints/level0_final.zip --no-spawn-sim
"""
import os
import sys
import time
import yaml
import signal
import atexit
import argparse
import subprocess
import numpy as np

# Allow direct import of kairos_rl before colcon install
PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

import rclpy
from stable_baselines3 import PPO

from kairos_rl.curriculum import ENVS


LEVEL_NAMES = {0: 'Reach', 1: 'Pick', 2: 'Place', 3: 'Full'}


def get_rl_xacro_path() -> str:
    """Dynamically resolve absolute path to rbkairos_ur5_rl.urdf.xacro."""
    try:
        from ament_index_python.packages import get_package_share_directory
        share_path = os.path.join(get_package_share_directory('rbkairos_description'), 'robots', 'rbkairos_ur5_rl.urdf.xacro')
        if os.path.exists(share_path):
            return share_path
    except Exception:
        pass
    candidates = [
        "/home/robot/ros2_ws/src/rbkairos_description/robots/rbkairos_ur5_rl.urdf.xacro",
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'rbkairos_description', 'robots', 'rbkairos_ur5_rl.urdf.xacro')),
        os.path.expanduser('~/ros2_ws/src/rbkairos_description/robots/rbkairos_ur5_rl.urdf.xacro'),
        os.path.expanduser('~/kairos_ws_rl/src/rbkairos_description/robots/rbkairos_ur5_rl.urdf.xacro'),
        "/home/YourMachine/kairos_ws_rl/src/rbkairos_description/robots/rbkairos_ur5_rl.urdf.xacro",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return "/home/robot/ros2_ws/src/rbkairos_description/robots/rbkairos_ur5_rl.urdf.xacro"


def load_config(config_path: str) -> dict:
    """Load YAML config with path fallback heuristics."""
    candidates = [
        config_path,
        os.path.join('src', 'kairos_rl', 'config', os.path.basename(config_path)),
    ]
    try:
        from ament_index_python.packages import get_package_share_directory
        share_dir = get_package_share_directory('kairos_rl')
        candidates.append(os.path.join(share_dir, 'config', os.path.basename(config_path)))
    except Exception:
        pass

    for candidate in candidates:
        if os.path.exists(candidate):
            print(f"[Config] Loading from: {candidate}", flush=True)
            with open(candidate, 'r') as f:
                return yaml.safe_load(f)

    print(f"⚠️  WARNING: File '{config_path}' not found, using default configuration.", flush=True)
    return {}


def is_ros_sim_running() -> bool:
    """Check if simulation topics are already active."""
    try:
        res = subprocess.run(["ros2", "topic", "list"], capture_output=True, text=True, timeout=5)
        return "/robot/joint_states" in res.stdout
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="RL Policy Evaluation for KAIROS Curriculum Learning"
    )
    parser.add_argument(
        '--level', type=int, default=0, choices=[0, 1, 2, 3],
        help='Curriculum level to evaluate (0=Reach, 1=Pick, 2=Place, 3=Full)'
    )
    parser.add_argument(
        '--checkpoint', type=str, default='./checkpoints/level0_final.zip',
        help='Path to checkpoint .zip file'
    )
    parser.add_argument(
        '--config', type=str, default='config/curriculum_params.yaml',
        help='YAML configuration file path'
    )
    parser.add_argument(
        '--episodes', type=int, default=10,
        help="Number of evaluation episodes"
    )
    parser.add_argument(
        '--max-steps', type=int, default=None,
        help='Maximum steps per episode (default: config value)'
    )
    parser.add_argument(
        '--gui', action='store_true', default=True,
        help='Enable Gazebo GUI window (default: enabled)'
    )
    parser.add_argument(
        '--no-gui', dest='gui', action='store_false',
        help='Disable Gazebo GUI window'
    )
    parser.add_argument(
        '--no-spawn-sim', action='store_true',
        help='Do not spawn Gazebo (connect to already running simulation)'
    )
    parser.add_argument(
        '--stochastic', action='store_true', default=False,
        help='Use stochastic actions instead of deterministic mean'
    )
    parser.add_argument(
        '--device', type=str, default='auto',
        help='PyTorch compute device (cpu, cuda, auto)'
    )

    args = parser.parse_args()

    # 1. Checkpoint path check
    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        alt_path = os.path.join('checkpoints', os.path.basename(checkpoint_path))
        if os.path.exists(alt_path):
            checkpoint_path = alt_path
        else:
            print(f"❌ Error: Checkpoint '{args.checkpoint}' not found!", file=sys.stderr)
            sys.exit(1)

    # 2. Configuration
    config = load_config(args.config)
    level = args.level
    level_name = LEVEL_NAMES[level]

    print("\n" + "=" * 65)
    print(f"  🔍 Evaluation KAIROS RL — Level {level}: {level_name}")
    print(f"  Model        : {checkpoint_path}")
    print(f"  Episodes     : {args.episodes}")
    print(f"  Action mode  : {'Stochastic' if args.stochastic else 'Deterministic'}")
    print(f"  GUI Display  : {args.gui}")
    print("=" * 65 + "\n", flush=True)

    # 3. Launch Gazebo if requested and not running
    procs = []
    if not args.no_spawn_sim:
        sim_already_up = is_ros_sim_running()
        if sim_already_up:
            print("ℹ️  Existing Gazebo simulation detected. Connecting directly...", flush=True)
        else:
            gui_str = "true" if args.gui else "false"
            print("🚀 Launching Gazebo Harmonic...", flush=True)
            world_cmd = [
                "ros2", "launch", "robotnik_gazebo_ignition", "spawn_world.launch.py",
                "world:=labo", f"gui:={gui_str}"
            ]
            env_vars = os.environ.copy()
            if args.gui:
                env_vars.setdefault('DISPLAY', ':0')
            p_world = subprocess.Popen(world_cmd, start_new_session=True, env=env_vars,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            procs.append(p_world)
            time.sleep(8)
            print("🌍 World ready. Spawning robot...", flush=True)

            robot_cmd = [
                "ros2", "launch", "robotnik_gazebo_ignition", "spawn_robot.launch.py",
                "gripper_type:=schunk_egk50",
                f"robot_xacro_path:={get_rl_xacro_path()}",
                "use_sim:=true", "gazebo_ignition:=true", "robot:=rbkairos", "robot_model:=rbkairos_plus",
                "run_rviz:=false"
            ]
            p_robot = subprocess.Popen(robot_cmd, start_new_session=True, env=env_vars,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            procs.append(p_robot)
            time.sleep(12)
            print("🤖 Robot ready.", flush=True)

    def cleanup():
        if procs:
            print("\n🧹 Cleaning simulation subprocesses...", flush=True)
            for p in procs:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                except Exception:
                    pass
            time.sleep(1)
            for p in procs:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                except Exception:
                    pass
    atexit.register(cleanup)

    # 4. Instantiate Gym environment
    EnvClass = ENVS[level]
    env = EnvClass(config=config)
    if args.max_steps is not None:
        env.max_steps = args.max_steps

    # 5. Load PPO policy
    print(f"📦 Loading PPO policy from: {checkpoint_path}...", flush=True)
    try:
        model = PPO.load(checkpoint_path, env=env, device=args.device)
        print("✅ Policy successfully loaded.", flush=True)
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}", file=sys.stderr)
        env.close()
        sys.exit(1)

    # 6. Evaluation loop
    episode_returns = []
    episode_lengths = []
    successes = []
    collisions = []

    print("\n" + "-" * 65)
    print(f"{'Episode':^9} | {'Score':^10} | {'Steps':^6} | {'Success':^8} | {'Collision':^10}")
    print("-" * 65)

    try:
        for ep in range(1, args.episodes + 1):
            obs, info = env.reset()
            done = False
            truncated = False
            ep_return = 0.0
            ep_steps = 0
            ep_success = False
            ep_collision = False

            while not (done or truncated):
                action, _states = model.predict(obs, deterministic=not args.stochastic)
                obs, reward, terminated, truncated, step_info = env.step(action)
                ep_return += reward
                ep_steps += 1
                done = terminated

                if step_info.get('success', False):
                    ep_success = True
                if step_info.get('self_collision', False) or step_info.get('collision', False):
                    ep_collision = True

            episode_returns.append(ep_return)
            episode_lengths.append(ep_steps)
            successes.append(1 if ep_success else 0)
            collisions.append(1 if ep_collision else 0)

            succ_str = "✅ YES" if ep_success else "❌ NO"
            coll_str = "⚠️ YES" if ep_collision else "—"
            print(f"{ep:^9} | {ep_return:^10.2f} | {ep_steps:^6} | {succ_str:^8} | {coll_str:^10}", flush=True)

    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted by user.")

    finally:
        env.close()

    # 7. Statistical summary report
    if episode_returns:
        n = len(episode_returns)
        mean_ret = np.mean(episode_returns)
        std_ret = np.std(episode_returns)
        mean_len = np.mean(episode_lengths)
        succ_rate = (sum(successes) / n) * 100.0
        coll_rate = (sum(collisions) / n) * 100.0

        print("\n" + "=" * 65)
        print("  📊 EVALUATION SUMMARY")
        print("=" * 65)
        print(f"  Episodes completed: {n} / {args.episodes}")
        print(f"  Mean Reward       : {mean_ret:.2f} ± {std_ret:.2f}")
        print(f"  Mean Episode Length: {mean_len:.1f} steps")
        print(f"  Success Rate      : {succ_rate:.1f} %")
        print(f"  Collision Rate    : {coll_rate:.1f} %")
        print("=" * 65 + "\n")


if __name__ == '__main__':
    main()
