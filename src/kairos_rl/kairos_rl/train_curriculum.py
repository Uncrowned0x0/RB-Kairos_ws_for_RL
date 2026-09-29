# ==============================================================================
# Author: Kamil BENMADI
# Email: kamil.benmadi@sigma-clermont.fr
# GitHub: https://github.com/Uncrowned0x0
# ==============================================================================

"""
Curriculum Learning Training Script for KAIROS Pick & Place.
Compatible with SubprocVecEnv for multi-instance parallelism.

Usage:
  # Level 0 (Reach)
  ros2 run kairos_rl train_curriculum --level 0 --num-envs 8 --total-timesteps 500000

  # Level 1 (Pick) with transfer learning
  ros2 run kairos_rl train_curriculum --level 1 --resume ./checkpoints/level0_final.zip --num-envs 8

  # Level 2 (Place)
  ros2 run kairos_rl train_curriculum --level 2 --resume ./checkpoints/level1_final.zip --num-envs 8

  # Level 3 (End-to-End)
  ros2 run kairos_rl train_curriculum --level 3 --resume ./checkpoints/level2_final.zip --num-envs 8
"""
import os
import sys
import time
import yaml
import glob
import argparse
import subprocess
import signal
import atexit

# Allow direct import of kairos_rl before colcon build/install
PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

import rclpy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    CallbackList,
)

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
    # 1. Check provided path
    if not os.path.exists(config_path):
        # 2. Look in src/ (convenient for development without colcon build)
        src_path = os.path.join('src', 'kairos_rl', 'config', os.path.basename(config_path))
        if os.path.exists(src_path):
            config_path = src_path
        else:
            # 3. Look in ROS 2 package share
            try:
                from ament_index_python.packages import get_package_share_directory
                share_dir = get_package_share_directory('kairos_rl')
                alt_path = os.path.join(share_dir, 'config', os.path.basename(config_path))
                if os.path.exists(alt_path):
                    config_path = alt_path
            except Exception:
                pass

    if os.path.exists(config_path):
        print(f"[Config] Loading parameters from: {config_path}", flush=True)
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
            
    print(f"⚠️  WARNING: Configuration file '{config_path}' not found! Using hardcoded defaults.", flush=True)
    return {}


class EnvWorkerFactory:
    """
    Callable factory for a Curriculum Learning environment.
    Pickle-serializable, compatible with SubprocVecEnv(start_method='spawn').
    """
    def __init__(self, rank: int, level: int, config: dict, gui_on_zero: bool = True):
        self.rank = rank
        self.level = level
        self.config = config
        self.gui_on_zero = gui_on_zero

    def __call__(self):
        rank = self.rank
        level = self.level
        config = self.config
        gui_on_zero = self.gui_on_zero

        # Stagger startup to avoid simultaneous spawn overload
        if rank > 0:
            time.sleep(rank * 1.5)

        # ROS 2 and Gazebo isolation
        domain_id = 10 + rank
        os.environ['ROS_DOMAIN_ID'] = str(domain_id)
        os.environ['GZ_PARTITION'] = f"sim{domain_id}"
        print(f"[Worker {rank}] 🌐 Level {level} ({LEVEL_NAMES[level]}) | "
              f"ROS_DOMAIN_ID={domain_id}", flush=True)

        # Spawn World (no LiDAR raycasting overhead)
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
        p_world = subprocess.Popen(world_cmd, start_new_session=True, env=env_vars,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(8)
        print(f"[Worker {rank}] 🌍 World ready.", flush=True)

        # Spawn Robot (without RViz to save GPU and CPU during training)
        robot_cmd = [
            "ros2", "launch", "robotnik_gazebo_ignition", "spawn_robot.launch.py",
            "gripper_type:=schunk_egk50",
            f"robot_xacro_path:={get_rl_xacro_path()}",
            "use_sim:=true", "gazebo_ignition:=true", "robot:=rbkairos", "robot_model:=rbkairos_plus",
            "run_rviz:=false"
        ]
        p_robot = subprocess.Popen(robot_cmd, start_new_session=True, env=env_vars,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(15)
        print(f"[Worker {rank}] 🤖 Robot ready.", flush=True)

        # Cleanup handler
        def cleanup():
            print(f"[Worker {rank}] Shutting down...", flush=True)
            for proc in [p_robot, p_world]:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except Exception:
                    pass
            time.sleep(2)
            for proc in [p_robot, p_world]:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    pass
        atexit.register(cleanup)

        # Instantiate environment for selected curriculum level
        EnvClass = ENVS[level]
        env = EnvClass(config=config)
        print(f"[Worker {rank}] ✅ Ready!", flush=True)
        return env


def make_env(rank: int, level: int, config: dict, gui_on_zero: bool = True):
    return EnvWorkerFactory(rank, level, config, gui_on_zero)


def main():
    parser = argparse.ArgumentParser(
        description='Curriculum Learning Training for KAIROS Mobile Manipulation'
    )
    parser.add_argument(
        '--level', type=int, required=True, choices=[0, 1, 2, 3],
        help='Curriculum level (0=Reach, 1=Pick, 2=Place, 3=Full)'
    )
    parser.add_argument(
        '--num-envs', type=int, default=2,
        help='Number of parallel environment workers'
    )
    parser.add_argument(
        '--config', type=str, default='config/curriculum_params.yaml',
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--total-timesteps', type=int, default=500_000,
        help='Total training timesteps'
    )
    parser.add_argument(
        '--checkpoint-freq', type=int, default=25_000,
        help='Checkpoint saving frequency (timesteps)'
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
        help='Path to existing PPO checkpoint for resuming or transfer learning'
    )
    args = parser.parse_args()

    # Configuration loading
    config = load_config(args.config)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    level_name = LEVEL_NAMES[args.level]
    print(f"\n{'='*60}")
    print(f"  🎯 Curriculum Learning — Level {args.level}: {level_name}")
    print(f"  Workers: {args.num_envs} | Timesteps: {args.total_timesteps:,}")
    print(f"{'='*60}\n")

    # FastRTPS shared memory cleanup
    try:
        for f in glob.glob("/dev/shm/fastrtps_*") + glob.glob("/dev/shm/sem.fastrtps_*"):
            try:
                os.remove(f)
            except Exception:
                pass
    except Exception:
        pass

    # Parallel environments (start_method='spawn' for clean ROS 2 isolation)
    env_fns = [make_env(i, args.level, config, gui_on_zero=True) for i in range(args.num_envs)]
    vec_env = SubprocVecEnv(env_fns, start_method='spawn')
    vec_env = VecMonitor(vec_env)

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq // args.num_envs,
        save_path=args.save_dir,
        name_prefix=f"kairos_level{args.level}_{level_name.lower()}"
    )
    callback = CallbackList([checkpoint_callback])

    # PPO configuration from YAML
    ppo_cfg = config.get('ppo', {})

    # Model initialization / loading
    if args.resume:
        print(f"📦 Loading weights from: {args.resume}")
        
        # Try direct loading (same level, resuming training)
        try:
            model = PPO.load(
                args.resume,
                env=vec_env,
                tensorboard_log=args.log_dir,
                learning_rate=ppo_cfg.get('learning_rate', 3e-4),
                n_steps=ppo_cfg.get('n_steps', 2048),
                batch_size=ppo_cfg.get('batch_size', 64),
                n_epochs=ppo_cfg.get('n_epochs', 10),
                gamma=ppo_cfg.get('gamma', 0.99),
                gae_lambda=ppo_cfg.get('gae_lambda', 0.95),
                clip_range=ppo_cfg.get('clip_range', 0.2),
                ent_coef=ppo_cfg.get('ent_coef', 0.01),
                device=ppo_cfg.get('device', 'cpu'),
            )
            print(f"✅ Weights loaded directly (compatible obs/action spaces).", flush=True)
        except ValueError as e:
            # Incompatible spaces -> Partial Transfer Learning
            print(f"⚠️  Incompatible observation/action spaces: {e}")
            print(f"🔄 Partial Transfer Learning: copying compatible layer weights...", flush=True)
            
            # 1. Load old model without environment
            old_model = PPO.load(args.resume, env=None, device='cpu')
            old_state = old_model.policy.state_dict()
            
            # 2. Create new model for current environment
            model = PPO(
                policy=ppo_cfg.get('policy', 'MlpPolicy'),
                env=vec_env,
                learning_rate=ppo_cfg.get('learning_rate', 3e-4),
                n_steps=ppo_cfg.get('n_steps', 2048),
                batch_size=ppo_cfg.get('batch_size', 64),
                n_epochs=ppo_cfg.get('n_epochs', 10),
                gamma=ppo_cfg.get('gamma', 0.99),
                gae_lambda=ppo_cfg.get('gae_lambda', 0.95),
                clip_range=ppo_cfg.get('clip_range', 0.2),
                ent_coef=ppo_cfg.get('ent_coef', 0.01),
                verbose=1,
                tensorboard_log=args.log_dir,
                device=ppo_cfg.get('device', 'cpu'),
            )
            new_state = model.policy.state_dict()
            
            # 3. Copy matching weights
            transferred = 0
            skipped = 0
            for key in new_state:
                if key in old_state and old_state[key].shape == new_state[key].shape:
                    new_state[key] = old_state[key]
                    transferred += 1
                else:
                    skipped += 1
            
            model.policy.load_state_dict(new_state)
            print(f"✅ Transfer Learning: {transferred} layers transferred, "
                  f"{skipped} layers initialized randomly.", flush=True)
    else:
        print(f"🆕 Initializing new PPO model for Level {args.level}.")
        model = PPO(
            policy=ppo_cfg.get('policy', 'MlpPolicy'),
            env=vec_env,
            learning_rate=ppo_cfg.get('learning_rate', 3e-4),
            n_steps=ppo_cfg.get('n_steps', 2048),
            batch_size=ppo_cfg.get('batch_size', 64),
            n_epochs=ppo_cfg.get('n_epochs', 10),
            gamma=ppo_cfg.get('gamma', 0.99),
            gae_lambda=ppo_cfg.get('gae_lambda', 0.95),
            clip_range=ppo_cfg.get('clip_range', 0.2),
            ent_coef=ppo_cfg.get('ent_coef', 0.01),
            verbose=1,
            tensorboard_log=args.log_dir,
            device=ppo_cfg.get('device', 'cpu'),
        )

    # Training execution
    print(f"\n🚀 Starting training for Level {args.level} ({level_name})...\n", flush=True)
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback,
        tb_log_name=f"level{args.level}_{level_name.lower()}",
    )

    # Final checkpoint save
    final_path = os.path.join(args.save_dir, f"level{args.level}_final")
    model.save(final_path)
    print(f"\n💾 Final model saved: {final_path}.zip")
    print(f"✅ Level {args.level} ({level_name}) training completed!")

    vec_env.close()


if __name__ == '__main__':
    main()
