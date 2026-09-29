"""
Level 2 — Place: Cube starts locked in the gripper.
The agent must move the arm to the drop target zone and release.

Action space (7D): Δq1..Δq6 + gripper_cmd
Observation space (20D):
    [0:6]   - Arm joint positions
    [6:12]  - Arm joint velocities
    [12]    - Gripper opening
    [13:16] - TCP → Drop zone vector (Δx, Δy, Δz)
    [16:19] - Cube position (x, y, z)
    [19]    - Grasp status
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


class PlaceEnv(gym.Env):
    """
    Level 2 — Place.
    The cube is already grasped. The agent deposits it into the target zone.
    """

    metadata = {'render_modes': []}

    def __init__(self, config: dict = None):
        super().__init__()

        if config is None:
            config = {}
        env_cfg = config.get('environment', {})
        self.control_hz = env_cfg.get('control_hz', 10)
        max_steps_cfg = env_cfg.get('max_steps_per_level', {})
        self.max_steps = max_steps_cfg.get('level2', 400)

        arm_cfg = config.get('arm', {})
        self.delta_q_max = arm_cfg.get('delta_q_max', 0.05)
        self.home_position = np.array(
            arm_cfg.get('home_position', HOME_POSITION.tolist())
        )

        cube_cfg = config.get('cube', {})
        self.cube_size_range = cube_cfg.get('size_range', [0.03, 0.055])
        self.cube_mass_values = cube_cfg.get('mass_values', [0.5, 1.0, 1.5, 2.0])
        self.spawn_z = cube_cfg.get('spawn_z', 0.76)

        drop_cfg = config.get('drop_zone', {})
        self.drop_position = np.array(
            drop_cfg.get('position', [0.4, -0.3, 0.90])
        )

        reward_cfg = config.get('reward', {})
        self.place_drop_threshold = reward_cfg.get('place_drop_threshold', 0.15)
        self.self_collision_penalty = reward_cfg.get('self_collision_penalty', -200.0)
        self.action_reg_coeff = reward_cfg.get('action_reg_coeff', 0.01)

        # ── Spaces ──
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32
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
        self._worker_id = int(os.environ.get('ROS_DOMAIN_ID', '10')) - 10
        self._step_count = 0
        self._total_episodes = 0
        self._total_successes = 0
        self._total_collisions = 0
        self._cube_name = 'rl_cube'
        self._cube_mass = 1.0
        self._cube_size = 0.04
        self._grasped = True  # Starts with the cube grasped
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

    # ─── Reset ────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._cube_size = random.uniform(*self.cube_size_range)
        self._cube_mass = random.choice(self.cube_mass_values)

        arm_pos, _, _, _ = self._ros.get_state()
        tcp_arm = ur5e_fk(arm_pos, tcp_offset=self.tcp_offset)
        
        # Intelligent physical clearance (Safe Home)
        if tcp_arm[2] < 0.15:
            safe_pos = arm_pos.copy()
            safe_pos[1] = -1.57
            safe_pos[2] = 0.0
            self._ros.publish_arm_target(safe_pos, duration_sec=1.0)
            self._spin_wait(1.0)

        # Initial arm position: slightly extended to simulate a grasp
        grasp_position = np.array([0.0, -1.2, -0.5, -1.3, 1.57, 0.0])
        self._ros.publish_arm_target(grasp_position, duration_sec=1.0)
        
        max_err = 100.0
        for _ in range(50):
            self._spin_wait(0.1)
            arm_pos, _, _, _ = self._ros.get_state()
            max_err = np.max(np.abs(arm_pos - grasp_position))
            if max_err < 0.02:
                break
        self._ros.publish_gripper(0.005)  # Nearly closed gripper
        self._spin_wait(0.2)

        # Teleport cube into gripper
        arm_pos, _, _, _ = self._ros.get_state()
        tcp_world = self._get_tcp_world(arm_pos)
        cube_x, cube_y, cube_z = tcp_world[0], tcp_world[1], tcp_world[2] - 0.02

        gz_utils.delete_model(self._cube_name)
        gz_utils.spawn_cube(self._cube_name, cube_x, cube_y, cube_z,
                            self._cube_size, self._cube_mass)
        self._cube_spawned = True
        self._spin_wait(0.5)

        with self._ros._lock:
            self._ros.cube_pos_world = np.array([cube_x, cube_y, cube_z])

        # Reset state
        self._step_count = 0
        self._grasped = True
        self._prev_dist = None
        self._ep_collisions = 0
        self._is_first_step = True

        obs = self._get_observation()
        info = {'cube_mass': self._cube_mass}
        print(f"[Place] Reset | Drop zone=({self.drop_position[0]:.2f}, "
              f"{self.drop_position[1]:.2f}, {self.drop_position[2]:.2f}) | "
              f"Mass={self._cube_mass}kg | MaxErr={max_err:.3f}rad", flush=True)
        return obs, info

    # ─── Step ─────────────────────────────────────────────────────────────────

    def step(self, action):
        self._step_count += 1
        action = np.clip(action, -1.0, 1.0)

        # Arm
        arm_pos, _, _, _ = self._ros.get_state()
        delta_q = action[:6] * self.delta_q_max
        target_q = np.clip(arm_pos + delta_q, ARM_LOWER, ARM_UPPER)
        self._ros.publish_arm_target(target_q, duration_sec=1.0 / self.control_hz)

        # Gripper
        gripper_cmd = float((action[6] + 1.0) / 2.0 * SCHUNK_MAX_OPEN)
        self._ros.publish_gripper(gripper_cmd)

        self._spin_wait(1.0 / self.control_hz)

        # State
        arm_pos, _, grip_pos, cube_pos = self._ros.get_state()
        tcp_world = self._get_tcp_world(arm_pos)

        # Release detection
        if self._grasped and gripper_cmd > 0.02:
            self._grasped = False

        obs = self._get_observation()
        reward, terminated, info = self._compute_reward(action, obs, cube_pos)

        truncated = self._step_count >= self.max_steps

        if terminated or truncated:
            self._total_episodes += 1
            is_succ = bool(info.get('success', False))
            if is_succ:
                self._total_successes += 1
                info['is_success'] = 1.0
            else:
                info['is_success'] = 0.0

            result = '✅ PLACED' if is_succ else '❌ Failed'
            if info.get('self_collision'):
                result = f'💥 COLLISION (R={reward:+.1f})'

            rate = (self._total_successes / self._total_episodes * 100.0) if self._total_episodes > 0 else 0.0
            print(f"[Worker {self._worker_id} - Place] End ep. {self._total_episodes} | Step {self._step_count}/{self.max_steps} | "
                  f"{result} | Success: {self._total_successes}/{self._total_episodes} ({rate:.1f}%) | "
                  f"Collisions: {self._total_collisions} | R={reward:+.2f}", flush=True)

        if self._step_count % 20 == 0:
            dist = float(np.linalg.norm(obs[13:16]))
            print(f"[Place] Step {self._step_count:03d} | DistDrop={dist:.3f}m | "
                  f"Grasped={self._grasped} | R={reward:+.2f}", flush=True)

        return obs, reward, terminated, truncated, info

    # ─── Observation ──────────────────────────────────────────────────────────

    def _get_observation(self):
        arm_pos, arm_vel, grip_pos, cube_pos = self._ros.get_state()
        tcp_world = self._get_tcp_world(arm_pos)

        tcp_to_drop = (self.drop_position - tcp_world).astype(np.float32)
        grasp_status = 1.0 if self._grasped else 0.0

        obs = np.concatenate([
            arm_pos.astype(np.float32),              # 6
            arm_vel.astype(np.float32),              # 6
            np.array([grip_pos], dtype=np.float32),  # 1
            tcp_to_drop,                             # 3
            cube_pos.astype(np.float32),             # 3
            np.array([grasp_status], dtype=np.float32),  # 1
        ]).astype(np.float32)

        return obs

    # ─── Reward ───────────────────────────────────────────────────────────────

    def _compute_reward(self, action, obs, cube_pos):
        info = {}
        terminated = False

        tcp_to_drop = obs[13:16]
        dist_to_drop = float(np.linalg.norm(tcp_to_drop))
        cube_to_drop = float(np.linalg.norm(cube_pos - self.drop_position))

        # Delta shaping towards drop zone
        r_approach = 0.0
        if self._prev_dist is not None:
            delta = self._prev_dist - dist_to_drop
            r_approach = delta * 50.0

        # Success: cube released in drop zone
        r_success = 0.0
        if not self._grasped and cube_to_drop < self.place_drop_threshold and cube_pos[2] > 0.5:
            r_success = 200.0
            terminated = True
            info['success'] = True

        # Penalty: cube dropped outside zone
        r_fell = 0.0
        if not self._grasped and cube_pos[2] < 0.05:
            r_fell = -50.0
            terminated = True
            info['cube_dropped'] = True

        # Self-collision (KAIROS chassis, crossing under table, arm self-collision)
        # Checked at each step to stop immediately on first contact
        r_collision = 0.0
        arm_pos, _, _, _ = self._ros.get_state()
        chassis_hit, table_hit, arm_hit, any_col, _ = check_robot_collisions(arm_pos, tcp_offset=self.tcp_offset)
        if any_col:
            r_collision = self.self_collision_penalty
            self._total_collisions += 1
            terminated = True
            info['self_collision'] = True

        # Regularization
        r_action = -float(self.action_reg_coeff * np.sum(action ** 2))

        reward = r_approach + r_success + r_fell + r_collision + r_action

        self._prev_dist = dist_to_drop
        info['r_approach'] = r_approach
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
