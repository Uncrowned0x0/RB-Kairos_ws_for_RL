"""
Parallel PPO Training Script for KAIROS Pick & Place environment.
Uses Stable-Baselines3 with SubprocVecEnv to accelerate experience collection.
"""
import os
import sys
import time
import yaml
import argparse
import subprocess
import signal
import atexit

import rclpy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.utils import set_random_seed

# Allow direct import of kairos_rl before colcon install
PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

from kairos_rl.kairos_pick_place_env import KairosPickPlaceEnv


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
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def make_env(rank: int, config: dict, gui_on_zero: bool = True):
    """
    Utility function to spawn Gazebo, ROS, and create the environment inside an isolated subprocess.
    """
    def _init():
        # Stagger startups to avoid simultaneous CPU/Disk spikes
        if rank > 0:
            time.sleep(rank * 1.5)

        # 1. ROS 2 and Gazebo isolation
        domain_id = 10 + rank
        os.environ['ROS_DOMAIN_ID'] = str(domain_id)
        os.environ['GZ_PARTITION'] = f"sim{domain_id}"
        print(f"[Worker {rank}] 🌐 Initializing (ROS_DOMAIN_ID={domain_id}, GZ_PARTITION=sim{domain_id})...", flush=True)

        # 2. Spawn World
        show_gui = gui_on_zero and (rank == 0)
        gui_str = "true" if show_gui else "false"
        
        world_cmd = [
            "ros2", "launch", "robotnik_gazebo_ignition", "spawn_world.launch.py", 
            "world:=labo", f"gui:={gui_str}"
        ]
        env_vars = os.environ.copy()
        if show_gui:
            env_vars.setdefault('DISPLAY', ':0')
            env_vars.setdefault('XAUTHORITY', os.path.expanduser('~/.Xauthority'))
        p_world = subprocess.Popen(world_cmd, start_new_session=True, env=env_vars, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        time.sleep(8)
        print(f"[Worker {rank}] 🌍 Gazebo World ready. Spawning KAIROS robot...", flush=True)

        # 3. Spawn Robot
        robot_cmd = [
            "ros2", "launch", "robotnik_gazebo_ignition", "spawn_robot.launch.py",
            "gripper_type:=schunk_egk50",
            f"robot_xacro_path:={get_rl_xacro_path()}",
            "use_sim:=true", "gazebo_ignition:=true", "robot:=rbkairos", "robot_model:=rbkairos_plus",
            "run_rviz:=false"
        ]
        p_robot = subprocess.Popen(robot_cmd, start_new_session=True, env=env_vars, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        time.sleep(15)
        print(f"[Worker {rank}] 🤖 KAIROS controllers active. Connecting RL...", flush=True)

        # 4. Clean shutdown handling
        def cleanup():
            print(f"[Worker {rank}] Shutting down Gazebo and ROS 2...", flush=True)
            try:
                os.killpg(os.getpgid(p_robot.pid), signal.SIGTERM)
            except Exception:
                pass
            try:
                os.killpg(os.getpgid(p_world.pid), signal.SIGTERM)
            except Exception:
                pass
            time.sleep(2)
            try:
                os.killpg(os.getpgid(p_robot.pid), signal.SIGKILL)
            except Exception:
                pass
            try:
                os.killpg(os.getpgid(p_world.pid), signal.SIGKILL)
            except Exception:
                pass

        atexit.register(cleanup)

        # 5. Create Gym environment
        env = KairosPickPlaceEnv(config=config)
        print(f"[Worker {rank}] ✅ Environment ready for collection!", flush=True)
        return env

    return _init


def main():
    parser = argparse.ArgumentParser(
        description='Parallel PPO Training for KAIROS Pick & Place'
    )
    parser.add_argument(
        '--num-envs', type=int, default=2,
        help='Number of parallel simulation instances'
    )
    parser.add_argument(
        '--config', type=str, default='config/rl_params.yaml',
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--total-timesteps', type=int, default=1_000_000,
        help='Total training timesteps'
    )
    parser.add_argument(
        '--checkpoint-freq', type=int, default=20_000,
        help='Checkpoint saving frequency'
    )
    parser.add_argument(
        '--log-dir', type=str, default='./tb_logs/',
        help='TensorBoard logging directory'
    )
    parser.add_argument(
        '--save-dir', type=str, default='./checkpoints/',
        help='Directory to store checkpoints'
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Path to existing PPO checkpoint to resume training'
    )
    args = parser.parse_args()

    # Configuration
    config = load_config(args.config)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    print(f"=== Launching Parallel Training ({args.num_envs} instances) ===")
    
    # FastDDS SHM cleanup
    try:
        import glob
        for f in glob.glob("/dev/shm/fastrtps_*") + glob.glob("/dev/shm/sem.fastrtps_*"):
            try:
                os.remove(f)
            except Exception:
                pass
    except Exception:
        pass

    # SubprocVecEnv environment pool
    env_fns = [make_env(i, config, gui_on_zero=True) for i in range(args.num_envs)]
    vec_env = SubprocVecEnv(env_fns, start_method='spawn')
    vec_env = VecMonitor(vec_env)

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq // args.num_envs,
        save_path=args.save_dir,
        name_prefix="kairos_ppo_multi"
    )
    callback = CallbackList([checkpoint_callback])

    # Model
    if args.resume:
        print(f"Resuming training from: {args.resume}")
        model = PPO.load(args.resume, env=vec_env)
    else:
        print("Initializing new PPO model...")
        ppo_cfg = config.get('ppo', {})
        model = PPO(
            policy=ppo_cfg.get('policy', 'MlpPolicy'),
            env=vec_env,
            learning_rate=float(ppo_cfg.get('learning_rate', 3e-4)),
            n_steps=ppo_cfg.get('n_steps', 2048),
            batch_size=ppo_cfg.get('batch_size', 64),
            n_epochs=ppo_cfg.get('n_epochs', 10),
            gamma=ppo_cfg.get('gamma', 0.99),
            gae_lambda=ppo_cfg.get('gae_lambda', 0.95),
            clip_range=ppo_cfg.get('clip_range', 0.2),
            ent_coef=ppo_cfg.get('ent_coef', 0.0),
            verbose=1,
            tensorboard_log=args.log_dir,
        )

    # Training loop
    try:
        print("Starting training...")
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback,
            reset_num_timesteps=(args.resume is None)
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        model.save(os.path.join(args.save_dir, "kairos_ppo_multi_final"))
        vec_env.close()
        print("Training cleanly terminated.")


if __name__ == "__main__":
    main()
