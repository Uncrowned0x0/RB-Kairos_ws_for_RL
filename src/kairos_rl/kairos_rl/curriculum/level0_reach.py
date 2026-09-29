# ==============================================================================
# Author: Kamil BENMADI
# Email: kamil.benmadi@sigma-clermont.fr
# GitHub: https://github.com/Uncrowned0x0
# ==============================================================================

"""
Level 0 — Reach: The UR5e arm learns to bring its TCP to a 3D XYZ target.
No physical object, no gripper. Pure Cartesian reaching geometry.

Action space (6D): Δq1..Δq6 (delta joint positions)
Observation space (15D):
    [0:6]   - Arm joint positions
    [6:12]  - Arm joint velocities
    [12:15] - TCP → Target vector (Δx, Δy, Δz) in arm_base_link frame
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

from kairos_rl.ros_interface import ArmRosInterface, ARM_JOINTS
from kairos_rl import gz_utils

# ─── UR5e DH Parameters ────────────────────────────────────────────────────────
_UR5E_DH = [
    (0.0,     0.1625,  math.pi / 2),
    (-0.4253, 0.0,     0.0),
    (-0.3922, 0.0,     0.0),
    (0.0,     0.1333,  math.pi / 2),
    (0.0,     0.0997, -math.pi / 2),
    (0.0,     0.0996,  0.0),
]

# UR5e joint limits (radians): restricted to [-pi, pi] to avoid unnecessary wrapping
ARM_LOWER = np.array([-math.pi, -math.pi, -math.pi, -math.pi, -math.pi, -math.pi])
ARM_UPPER = np.array([ math.pi,  math.pi,  math.pi,  math.pi,  math.pi,  math.pi])

HOME_POSITION = np.array([0.0, -1.57, 0.0, -1.57, 0.0, 0.0])

# Arm mount height on KAIROS base
ARM_MOUNT_Z = 0.84


def _dh_matrix(a, d, alpha, theta):
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([
        [ct,  -st * ca,  st * sa, a * ct],
        [st,   ct * ca, -ct * sa, a * st],
        [0.0,       sa,       ca,      d],
        [0.0,      0.0,      0.0,    1.0],
    ], dtype=np.float64)


# Transformation between robot_arm_base_link (REP-103, X+ forward) and robot_arm_base_link_inertia
# (internal UR / DH frame, X+ backward) defined in ur_macro.xacro: rpy="0 0 pi"
_T_BASE_TO_INERTIA = np.array([
    [-1.0,  0.0, 0.0, 0.0],
    [ 0.0, -1.0, 0.0, 0.0],
    [ 0.0,  0.0, 1.0, 0.0],
    [ 0.0,  0.0, 0.0, 1.0],
], dtype=np.float64)


def ur5e_fk(q, tcp_offset=None):
    """
    Complete UR5e FK -> TCP position in robot_arm_base_link frame (REP-103).
    If tcp_offset is provided ([dx, dy, dz] in tool0 frame), projects TCP
    to tool end-effector (e.g. center of gripper fingers).
    """
    T = _T_BASE_TO_INERTIA.copy()
    for i, (a, d, alpha) in enumerate(_UR5E_DH):
        T = T @ _dh_matrix(a, d, alpha, q[i])
    if tcp_offset is not None:
        offset = np.array(tcp_offset, dtype=np.float64)
        return T[:3, 3] + T[:3, :3] @ offset
    return T[:3, 3]


def _point_to_segment_distance(p, a, b):
    ab = b - a
    ap = p - a
    norm_ab2 = np.dot(ab, ab)
    if norm_ab2 == 0.0:
        return np.linalg.norm(ap)
    t = np.clip(np.dot(ap, ab) / norm_ab2, 0.0, 1.0)
    return np.linalg.norm(p - (a + t * ab))


def ur5e_self_collision_check(q, tcp_pos, threshold=0.08):
    """
    Geometrically check if TCP collides with the UR5e upper_arm_link.
    Returns True if TCP is closer than threshold (meters) to [shoulder, elbow] segment.
    Physical threshold: ~0.08m (arm radius 0.05m + gripper radius 0.03m).
    """
    T = _T_BASE_TO_INERTIA.copy()
    links = []
    for i, (a, d, alpha) in enumerate(_UR5E_DH):
        T = T @ _dh_matrix(a, d, alpha, q[i])
        links.append(T[:3, 3])
    
    shoulder = links[0]  # Joint 1 output
    elbow = links[1]     # Joint 2 output
    
    # Distance between TCP point and 3D segment [shoulder, elbow]
    dist = _point_to_segment_distance(tcp_pos, shoulder, elbow)
    return dist < threshold


def check_robot_collisions(q, threshold_arm=0.08, tcp_offset=None):
    """
    Geometrically check all potential UR5e collisions on the KAIROS:
    1. KAIROS chassis: bounding box in arm_base_link frame
       X in [-0.73, 0.41], Y in [-0.37, 0.37], Z < 0.05
    2. Work table: Z < -0.085
    3. UR5e self-collision: distance TCP <-> upper_arm_link segment

    Returns:
        chassis_hit (bool), table_hit (bool), arm_hit (bool), any_collision (bool), tcp (np.ndarray)
    """
    T = _T_BASE_TO_INERTIA.copy()
    links = [np.array([0.0, 0.0, 0.0])]
    for i, (a, d, alpha) in enumerate(_UR5E_DH):
        T = T @ _dh_matrix(a, d, alpha, q[i])
        links.append(T[:3, 3])

    tcp_flange = links[-1]
    pts = [
        links[2],                      # elbow
        0.5 * (links[1] + links[2]),   # mid upper arm
        0.5 * (links[2] + links[3]),   # mid forearm
        links[3],                      # wrist 1
        links[4],                      # wrist 2
        links[5],                      # wrist 3
        tcp_flange,                    # tool0 flange
    ]

    # If gripper is mounted, add fingertips and gripper center
    if tcp_offset is not None:
        offset = np.array(tcp_offset, dtype=np.float64)
        tcp_gripper = tcp_flange + T[:3, :3] @ offset
        pts.append(tcp_gripper)
        pts.append(0.5 * (tcp_flange + tcp_gripper))
        tcp_check = tcp_gripper
    else:
        tcp_check = tcp_flange

    # 1. KAIROS chassis (bounding box centered on robot_base_link)
    chassis_hit = any(-0.73 <= p[0] <= 0.41 and -0.37 <= p[1] <= 0.37 and p[2] < 0.05 for p in pts)

    # 2. Work table (surface at Z_arm = -0.085m)
    table_hit = any(p[2] < -0.085 for p in pts)

    # 3. Self-collision: tool distance to [shoulder, elbow] segment
    shoulder, elbow = links[1], links[2]
    dist_arm = _point_to_segment_distance(tcp_check, shoulder, elbow)
    arm_hit = dist_arm < threshold_arm

    any_collision = chassis_hit or table_hit or arm_hit
    return chassis_hit, table_hit, arm_hit, any_collision, tcp_check


class ReachEnv(gym.Env):
    """
    Level 0 — Reach.
    The arm learns to reach random XYZ coordinates with its TCP.
    """

    metadata = {'render_modes': []}

    def __init__(self, config: dict = None):
        super().__init__()

        if config is None:
            config = {}
        env_cfg = config.get('environment', {})
        self.control_hz = env_cfg.get('control_hz', 10)
        max_steps_cfg = env_cfg.get('max_steps_per_level', {})
        self.max_steps = max_steps_cfg.get('level0', 200)

        arm_cfg = config.get('arm', {})
        self.delta_q_max = arm_cfg.get('delta_q_max', 0.05)
        self.home_position = np.array(
            arm_cfg.get('home_position', HOME_POSITION.tolist())
        )

        reward_cfg = config.get('reward', {})
        self.reach_threshold = reward_cfg.get('reach_success_threshold', 0.02)
        self.collision_penalty = reward_cfg.get('collision_penalty', -50.0)
        self.self_collision_penalty = reward_cfg.get('self_collision_penalty', -200.0)
        self.action_reg_coeff = reward_cfg.get('action_reg_coeff', 0.01)

        # ── Spaces ──
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32
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

        # ── Episode state ──
        domain_id_str = os.environ.get('ROS_DOMAIN_ID', '0')
        self._worker_id = 0 if domain_id_str == '0' else int(domain_id_str) - 10
        self._step_count = 0
        self._target_pos = np.array([0.4, 0.0, 1.0])  # world frame
        self._prev_dist = None
        self._total_episodes = 0
        self._total_successes = 0
        self._total_collisions = 0
        self._marker_spawned = False
        self._marker_name = 'rl_target_marker'

        self._spin_wait(1.5)

    def _spin_wait(self, duration_sec):
        end = self._ros.get_clock().now() + rclpy.duration.Duration(seconds=duration_sec)
        while rclpy.ok() and self._ros.get_clock().now() < end:
            rclpy.spin_once(self._ros, timeout_sec=0.01)

    def _get_tcp_arm(self, arm_pos):
        """TCP in arm_base_link frame."""
        return ur5e_fk(arm_pos).astype(np.float32)

    def _sample_target(self):
        """Sample a random XYZ target in the reachable frontal workspace of UR5e."""
        radius = random.uniform(0.30, 0.70)
        angle = random.uniform(-1.2, 1.2)  # radians (frontal half-space: not behind robot)
        height = random.uniform(0.10, 0.50)  # Z from ~0.94m to ~1.34m (above chassis)

        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        z = ARM_MOUNT_Z + height

        return np.array([x, y, z], dtype=np.float32)

    # ─── Reset ────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        arm_pos, _, _, _ = self._ros.get_state()
        tcp_arm = self._get_tcp_arm(arm_pos)
        
        # Intelligent physical clearance (Safe Home)
        # If arm is very low (near chassis or table), raise shoulder first
        if tcp_arm[2] < 0.15:
            safe_pos = arm_pos.copy()
            safe_pos[1] = -1.57  # shoulder_lift upwards
            safe_pos[2] = 0.0    # elbow extended
            self._ros.publish_arm_target(safe_pos, duration_sec=1.0)
            self._spin_wait(1.0)

        # Move arm to rest home position
        self._ros.publish_arm_target(self.home_position, duration_sec=1.0)
        
        # Wait for joints to reach home_position within +-0.02 rad
        max_err = 100.0
        for _ in range(50):  # max 5.0s simulated
            self._spin_wait(0.1)
            arm_pos, _, _, _ = self._ros.get_state()
            max_err = np.max(np.abs(arm_pos - self.home_position))
            if max_err < 0.02:
                break

        # Generate new target
        self._target_pos = self._sample_target()

        # Visual marker (green sphere) in Gazebo (Worker 0 / GUI only)
        domain_id = os.environ.get('ROS_DOMAIN_ID', '0')
        if domain_id in ['0', '10']:
            tx, ty, tz = float(self._target_pos[0]), float(self._target_pos[1]), float(self._target_pos[2])
            if not self._marker_spawned:
                gz_utils.spawn_target_marker(self._marker_name, tx, ty, tz)
                self._marker_spawned = True
            else:
                gz_utils.set_model_pose(self._marker_name, tx, ty, tz)

        # Reset state
        self._step_count = 0
        self._prev_dist = None
        self._is_first_step = True

        obs = self._get_observation()
        info = {'target': self._target_pos.tolist()}
        print(f"[Reach] Reset | Target=({self._target_pos[0]:.2f}, "
              f"{self._target_pos[1]:.2f}, {self._target_pos[2]:.2f}) | "
              f"MaxErr={max_err:.3f}rad", flush=True)
        return obs, info

    # ─── Step ─────────────────────────────────────────────────────────────────

    def step(self, action):
        self._step_count += 1
        action = np.clip(action, -1.0, 1.0)

        # Appliquer le delta articulaire
        arm_pos, arm_vel, _, _ = self._ros.get_state()
        delta_q = action * self.delta_q_max
        target_q = np.clip(arm_pos + delta_q, ARM_LOWER, ARM_UPPER)
        self._ros.publish_arm_target(target_q, duration_sec=1.0 / self.control_hz)

        # Attendre la physique
        self._spin_wait(1.0 / self.control_hz)

        # Observer
        obs = self._get_observation()

        # Reward
        reward, terminated, info = self._compute_reward(action, obs)

        # Truncation
        truncated = self._step_count >= self.max_steps

        if terminated or truncated:
            self._total_episodes += 1
            is_succ = bool(info.get('success', False))
            if is_succ:
                self._total_successes += 1
                info['is_success'] = 1.0
            else:
                info['is_success'] = 0.0

            result = '✅ SUCCESS' if is_succ else '❌ Timeout/Fail'
            if info.get('self_collision', False):
                result = f'💥 COLLISION (R={reward:+.1f})'
            
            arm_pos, _, _, _ = self._ros.get_state()
            tcp_arm = self._get_tcp_arm(arm_pos)
            tcp_world = np.array([tcp_arm[0], tcp_arm[1], ARM_MOUNT_Z + tcp_arm[2]], dtype=np.float32)
            dist = float(np.linalg.norm(tcp_world - self._target_pos))
            
            rate = (self._total_successes / self._total_episodes * 100.0) if self._total_episodes > 0 else 0.0
            print(f"[Worker {self._worker_id} - Reach] End ep. {self._total_episodes} | Step {self._step_count}/{self.max_steps} | "
                  f"{result} | Success: {self._total_successes}/{self._total_episodes} ({rate:.1f}%) | "
                  f"Collisions: {self._total_collisions} | Dist={dist:.3f}m | R={reward:+.2f}", flush=True)

        # Periodic log
        if self._step_count % 20 == 0:
            dist = float(np.linalg.norm(obs[12:15]))
            print(f"[Reach] Step {self._step_count:03d} | Dist={dist:.3f}m | R={reward:+.2f}", flush=True)

        return obs, reward, terminated, truncated, info

    # ─── Observation ──────────────────────────────────────────────────────────

    def _get_observation(self):
        arm_pos, arm_vel, _, _ = self._ros.get_state()

        # TCP in arm_base_link frame
        tcp_arm = self._get_tcp_arm(arm_pos)

        # TCP in world frame (base fixed at origin, arm mounted at ARM_MOUNT_Z)
        tcp_world = np.array([tcp_arm[0], tcp_arm[1], ARM_MOUNT_Z + tcp_arm[2]], dtype=np.float32)
        tcp_to_target = (self._target_pos - tcp_world).astype(np.float32)

        obs = np.concatenate([
            arm_pos.astype(np.float32),       # 6
            arm_vel.astype(np.float32),       # 6
            tcp_to_target,                     # 3
        ]).astype(np.float32)

        return obs

    # ─── Reward ───────────────────────────────────────────────────────────────

    def _compute_reward(self, action, obs):
        info = {}
        terminated = False

        tcp_to_target = obs[12:15]
        dist = float(np.linalg.norm(tcp_to_target))

        # Dense reward: distance penalty
        r_dist = -dist

        # Delta shaping: reward for getting closer
        r_delta = 0.0
        if self._prev_dist is not None:
            delta = self._prev_dist - dist
            r_delta = delta * 50.0

        # Success
        r_success = 0.0
        if dist < self.reach_threshold:
            r_success = 100.0
            terminated = True
            info['success'] = True

        # Self-collision (KAIROS chassis, table collision, arm self-collision)
        # Checked every step to stop immediately on first contact
        r_collision = 0.0
        arm_pos, _, _, _ = self._ros.get_state()
        chassis_hit, table_hit, arm_hit, any_col, _ = check_robot_collisions(arm_pos)

        if any_col:
            r_collision = self.self_collision_penalty
            self._total_collisions += 1
            terminated = True
            info['self_collision'] = True

        # Action regularization
        r_action = -float(self.action_reg_coeff * np.sum(action ** 2))

        reward = r_dist + r_delta + r_success + r_collision + r_action

        self._prev_dist = dist
        info['dist'] = dist
        info['r_dist'] = r_dist
        info['r_delta'] = r_delta
        info['r_success'] = r_success

        return reward, terminated, info

    def close(self):
        try:
            self._executor.shutdown()
            if rclpy.ok():
                self._ros.destroy_node()
        except Exception:
            pass
