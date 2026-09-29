"""
Level 1 — Pick: Arm + gripper learn to grasp a cube.
Uses pretrained Level 0 weights for the reach/approach phase.

Action space (7D): Δq1..Δq6 + gripper_cmd
Observation space (17D):
    [0:6]   - Arm joint positions
    [6:12]  - Arm joint velocities
    [12]    - Gripper opening
    [13:16] - TCP → Cube vector (Δx, Δy, Δz)
    [16]    - Grasp status (0.0 or 1.0)
"""
import os
import math
import random
import threading
import numpy as np

import gymnasium as gym
from gymnasium import spaces

import rclpy
import rclpy.duration
from rclpy.executors import MultiThreadedExecutor

from kairos_rl.ros_interface import ArmRosInterface, SCHUNK_MAX_OPEN
from kairos_rl.curriculum.level0_reach import (
    ur5e_fk, ur5e_self_collision_check, check_robot_collisions,
    ARM_LOWER, ARM_UPPER, HOME_POSITION, ARM_MOUNT_Z,
)
from kairos_rl import gz_utils


class PickEnv(gym.Env):
    """
    Level 1 — Pick.
    The arm approaches the cube, the gripper grasps it, and lifts it.
    """

    metadata = {'render_modes': []}

    def __init__(self, config: dict = None):
        super().__init__()

        if config is None:
            config = {}
        env_cfg = config.get('environment', {})
        self.control_hz = env_cfg.get('control_hz', 10)
        max_steps_cfg = env_cfg.get('max_steps_per_level', {})
        self.max_steps = max_steps_cfg.get('level1', 400)

        arm_cfg = config.get('arm', {})
        self.delta_q_max = arm_cfg.get('delta_q_max', 0.05)
        self.home_position = np.array(
            arm_cfg.get('home_position', HOME_POSITION.tolist())
        )

        cube_cfg = config.get('cube', {})
        self.cube_size_range = cube_cfg.get('size_range', [0.03, 0.055])
        self.cube_mass_values = cube_cfg.get('mass_values', [0.5, 1.0, 1.5, 2.0])
        self.spawn_radius_range = cube_cfg.get('spawn_radius_range', [0.30, 0.55])
        self.spawn_angle_range = cube_cfg.get('spawn_angle_range', [-1.2, 1.2])
        self.spawn_z = cube_cfg.get('spawn_z', 0.76)

        reward_cfg = config.get('reward', {})
        self.pick_lift_threshold = reward_cfg.get('pick_lift_threshold', 0.10)
        self.collision_penalty = reward_cfg.get('collision_penalty', -50.0)
        self.self_collision_penalty = reward_cfg.get('self_collision_penalty', -200.0)
        self.action_reg_coeff = reward_cfg.get('action_reg_coeff', 0.01)

        # ── Spaces ──
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32
        )

        # ── ROS 2 ──
        if not rclpy.ok():
            rclpy.init()
        self._ros = ArmRosInterface()
        self._executor = MultiThreadedExecutor(num_threads=2)
        self._executor.add_node(self._ros)
        self._spin_thread = threading.Thread(
            target=self._executor.spin, daemon=True
        )
        self._spin_thread.start()

        gripper_cfg = config.get('gripper', {})
        self.tcp_offset = np.array(gripper_cfg.get('tcp_offset', [0.0, 0.0, 0.198]), dtype=np.float64)

        # ── Episode state ──
        self._step_count = 0
        self._worker_id = int(os.environ.get('ROS_DOMAIN_ID', '10')) - 10
        self._total_episodes = 0
        self._total_successes = 0
        self._total_collisions = 0
        self._cube_name = 'rl_cube'
        self._cube_mass = 1.0
        self._cube_size = 0.04
        self._cube_spawn_pos = np.array([0.4, 0.0, 0.78])
        self._grasped = False
        self._grasped_prev = False
        self._cube_spawned = False
        self._prev_dist = None

        self._spin_wait(1.5)

    def _spin_wait(self, duration_sec):
        end = self._ros.get_clock().now() + rclpy.duration.Duration(seconds=duration_sec)
        while rclpy.ok() and self._ros.get_clock().now() < end:
            rclpy.spin_once(self._ros, timeout_sec=0.01)

    def _get_tcp_world(self, arm_pos):
        """Effective TCP (grasp center) in the world frame."""
        tcp_arm = ur5e_fk(arm_pos, tcp_offset=self.tcp_offset)
        return np.array([tcp_arm[0], tcp_arm[1], ARM_MOUNT_Z + tcp_arm[2]], dtype=np.float32)

    def _spawn_cube_in_workspace(self):
        """Generate a random cube placed strictly on the training tables."""
        self._cube_size = random.uniform(*self.cube_size_range)
        self._cube_mass = random.choice(self.cube_mass_values)

        x, y = gz_utils.sample_cube_on_table(
            table_z=self.spawn_z,
            min_reach=self.spawn_radius_range[0],
            max_reach=self.spawn_radius_range[1],
        )
        z = self.spawn_z + self._cube_size / 2.0 + 0.005  # Gently lands on the table

        self._cube_spawn_pos = np.array([x, y, z])
        return x, y, z

    # ─── Reset ────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        arm_pos, _, _, _ = self._ros.get_state()
        tcp_arm = ur5e_fk(arm_pos, tcp_offset=self.tcp_offset)
        
        # Intelligent physical clearance (Safe Home)
        if tcp_arm[2] < 0.15:
            safe_pos = arm_pos.copy()
            safe_pos[1] = -1.57  # shoulder_lift upwards
            safe_pos[2] = 0.0    # elbow extended
            self._ros.publish_arm_target(safe_pos, duration_sec=1.0)
            self._spin_wait(1.0)

        # Return arm to home position
        self._ros.publish_arm_target(self.home_position, duration_sec=1.0)
        
        # Block reset until /joint_states confirms home position
        max_err = 100.0
        for _ in range(50):
            self._spin_wait(0.1)
            arm_pos, _, _, _ = self._ros.get_state()
            max_err = np.max(np.abs(arm_pos - self.home_position))
            if max_err < 0.02:
                break
        self._ros.publish_gripper(SCHUNK_MAX_OPEN)
        self._spin_wait(0.1)

        # Cube: preliminary deletion to prevent Gazebo EntityFactory rejection
        x, y, z = self._spawn_cube_in_workspace()
        gz_utils.delete_model(self._cube_name)
        gz_utils.spawn_cube(self._cube_name, x, y, z,
                            self._cube_size, self._cube_mass)
        self._cube_spawned = True
        self._spin_wait(0.5)

        with self._ros._lock:
            self._ros.cube_pos_world = self._cube_spawn_pos.copy()

        # Reset state
        self._step_count = 0
        self._grasped = False
        self._grasped_prev = False
        self._prev_dist = None
        self._is_first_step = True

        obs = self._get_observation()
        info = {'cube_mass': self._cube_mass, 'cube_size': self._cube_size}
        print(f"[Pick] Reset | Cube=({x:.2f}, {y:.2f}, {z:.2f}) | "
              f"Mass={self._cube_mass}kg | Size={self._cube_size*100:.1f}cm | "
              f"MaxErr={max_err:.3f}rad", flush=True)
        return obs, info

    # ─── Step ─────────────────────────────────────────────────────────────────

    def step(self, action):
        self._step_count += 1
        action = np.clip(action, -1.0, 1.0)

        # Arm: joint delta
        arm_pos, _, _, _ = self._ros.get_state()
        delta_q = action[:6] * self.delta_q_max
        target_q = np.clip(arm_pos + delta_q, ARM_LOWER, ARM_UPPER)
        self._ros.publish_arm_target(target_q, duration_sec=1.0 / self.control_hz)

        # Gripper: [-1, 1] -> [0, SCHUNK_MAX_OPEN]
        gripper_cmd = float((action[6] + 1.0) / 2.0 * SCHUNK_MAX_OPEN)
        self._ros.publish_gripper(gripper_cmd)

        # Wait for physics
        self._spin_wait(1.0 / self.control_hz)

        # State
        arm_pos, _, grip_pos, cube_pos = self._ros.get_state()
        tcp_world = self._get_tcp_world(arm_pos)

        # Grasp detection
        if not self._grasped:
            dist_to_cube = np.linalg.norm(tcp_world - cube_pos)
            cube_lifted = cube_pos[2] > (self.spawn_z + 0.05)
            if dist_to_cube < 0.05 and gripper_cmd < 0.02 and cube_lifted:
                self._grasped = True
        if self._grasped and gripper_cmd > 0.025:
            self._grasped = False

        obs = self._get_observation()
        reward, terminated, info = self._compute_reward(action, obs, cube_pos, tcp_world)

        truncated = self._step_count >= self.max_steps

        if terminated or truncated:
            self._total_episodes += 1
            is_succ = bool(info.get('success', False))
            if is_succ:
                self._total_successes += 1
                info['is_success'] = 1.0
            else:
                info['is_success'] = 0.0

            result = '✅ GRASPED+LIFTED' if is_succ else '❌ Failed'
            if info.get('self_collision'):
                result = f'💥 COLLISION (R={reward:+.1f})'

            rate = (self._total_successes / self._total_episodes * 100.0) if self._total_episodes > 0 else 0.0
            print(f"[Worker {self._worker_id} - Pick] End ep. {self._total_episodes} | Step {self._step_count}/{self.max_steps} | "
                  f"{result} | Success: {self._total_successes}/{self._total_episodes} ({rate:.1f}%) | "
                  f"Collisions: {self._total_collisions} | Grasped: {self._grasped} | R={reward:+.2f}", flush=True)

        if self._step_count % 20 == 0:
            dist = float(np.linalg.norm(obs[13:16]))
            print(f"[Pick] Step {self._step_count:03d} | Dist={dist:.3f}m | "
                  f"Grip={grip_pos:.3f} | Grasped={self._grasped} | R={reward:+.2f}", flush=True)

        self._grasped_prev = self._grasped
        return obs, reward, terminated, truncated, info

    # ─── Observation ──────────────────────────────────────────────────────────

    def _get_observation(self):
        arm_pos, arm_vel, grip_pos, cube_pos = self._ros.get_state()
        tcp_world = self._get_tcp_world(arm_pos)

        tcp_to_cube = (cube_pos - tcp_world).astype(np.float32)
        grasp_status = 1.0 if self._grasped else 0.0

        obs = np.concatenate([
            arm_pos.astype(np.float32),              # 6
            arm_vel.astype(np.float32),              # 6
            np.array([grip_pos], dtype=np.float32),  # 1
            tcp_to_cube,                             # 3
            np.array([grasp_status], dtype=np.float32),  # 1
        ]).astype(np.float32)

        return obs

    # ─── Reward ───────────────────────────────────────────────────────────────

    def _compute_reward(self, action, obs, cube_pos, tcp_world):
        info = {}
        terminated = False

        tcp_to_cube = obs[13:16]
        dist = float(np.linalg.norm(tcp_to_cube))

        r_approach = 0.0
        r_grasp = 0.0
        r_hold = 0.0
        r_success = 0.0
        r_fell = 0.0
        r_collision = 0.0
        r_action = 0.0

        # Approach phase
        if not self._grasped:
            if self._prev_dist is not None:
                delta = self._prev_dist - dist
                r_approach = delta * 100.0

        # Grasp phase (one-shot bonus)
        if self._grasped and not self._grasped_prev:
            r_grasp = 50.0
            info['grasped'] = True

        # Hold phase
        if self._grasped:
            r_hold = 5.0

        # Success: cube lifted above the table
        if self._grasped and cube_pos[2] > (self.spawn_z + self.pick_lift_threshold):
            r_success = 100.0
            terminated = True
            info['success'] = True

        # Cube dropped on the floor
        if cube_pos[2] < 0.05:
            r_fell = -30.0
            terminated = True
            info['cube_dropped'] = True

        # Self-collision (KAIROS chassis, crossing under table, arm self-collision)
        # Checked at each step to stop immediately on first contact
        arm_pos, _, _, _ = self._ros.get_state()
        chassis_hit, table_hit, arm_hit, any_col, _ = check_robot_collisions(arm_pos, tcp_offset=self.tcp_offset)
        if any_col:
            r_collision = self.self_collision_penalty
            self._total_collisions += 1
            terminated = True
            info['self_collision'] = True

        # Regularization
        r_action = -float(self.action_reg_coeff * np.sum(action ** 2))

        reward = r_approach + r_grasp + r_hold + r_success + r_fell + r_collision + r_action

        self._prev_dist = dist
        info['r_approach'] = r_approach
        info['r_grasp'] = r_grasp
        info['r_success'] = r_success

        return reward, terminated, info

    def close(self):
        gz_utils.delete_model(self._cube_name)
        try:
            self._executor.shutdown()
            if rclpy.ok():
                self._ros.destroy_node()
        except Exception:
            pass
