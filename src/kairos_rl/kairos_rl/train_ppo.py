"""
PPO Training Script for KAIROS Pick & Place environment.
Uses Stable-Baselines3 with TensorBoard logging.
"""
import os
import sys
import yaml
import argparse

# Allow direct import of kairos_rl before colcon install
PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

import rclpy
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor

from kairos_rl.kairos_pick_place_env import KairosPickPlaceEnv


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(
        description='PPO Training for KAIROS Pick & Place'
    )
    parser.add_argument(
        '--config', type=str,
        default='config/rl_params.yaml',
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--total-timesteps', type=int,
        default=500_000,
        help='Total training timesteps'
    )
    parser.add_argument(
        '--checkpoint-freq', type=int,
        default=10_000,
        help='Checkpoint saving frequency'
    )
    parser.add_argument(
        '--log-dir', type=str,
        default='./tb_logs/',
        help='TensorBoard logging directory'
    )
    parser.add_argument(
        '--save-dir', type=str,
        default='./checkpoints/',
        help='Directory to store checkpoints'
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Path to existing PPO checkpoint to resume training'
    )
    parser.add_argument(
        '--device', type=str, default='cpu',
        help='PyTorch compute device (cpu, cuda, auto)'
    )
    args = parser.parse_args()

    # ── Configuration ──
    config = load_config(args.config)

    # ── Environment ──
    env = KairosPickPlaceEnv(config=config)
    env = Monitor(env)

    # ── PPO Model ──
    if args.resume:
        print(f"Resuming training from: {args.resume}")
        model = PPO.load(args.resume, env=env, device=args.device)
    else:
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=args.log_dir,
            device=args.device,
        )

    # ── Callbacks ──
    os.makedirs(args.save_dir, exist_ok=True)

    checkpoint_cb = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=args.save_dir,
        name_prefix='kairos_ppo',
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    callbacks = CallbackList([checkpoint_cb])

    # ── Training ──
    print("=" * 60)
    print("  KAIROS Pick & Place — PPO Training")
    print(f"  Total timesteps : {args.total_timesteps}")
    print(f"  Checkpoint freq : {args.checkpoint_freq}")
    print(f"  TensorBoard     : {args.log_dir}")
    print(f"  Checkpoints     : {args.save_dir}")
    print("=" * 60)

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            progress_bar=False,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        # Final checkpoint save
        final_path = os.path.join(args.save_dir, 'kairos_ppo_final')
        model.save(final_path)
        print(f"Final model saved: {final_path}")
        env.close()


if __name__ == '__main__':
    main()
