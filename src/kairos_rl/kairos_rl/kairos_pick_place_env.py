"""
Gymnasium environment for Pick & Place with KAIROS + UR5e + Schunk.
Compatible with Stable-Baselines3 (PPO).
"""
import rclpy.duration
import math
import random
import threading
import numpy as np

import gymnasium as gym
from gymnasium import spaces

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import Twist, TwistStamped, Pose
from sensor_msgs.msg import JointState, LaserScan
from nav_msgs.msg import Odometry
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray

from kairos_rl import gz_utils

# ─── Joint names ───────────────────────────────────────────────────────────────
ARM_JOINTS = [
    'robot_arm_shoulder_pan_joint',
    'robot_arm_shoulder_lift_joint',
    'robot_arm_elbow_joint',
    'robot_arm_wrist_1_joint',
    'robot_arm_wrist_2_joint',
    'robot_arm_wrist_3_joint',
]

# UR5e joint limits (radians)
ARM_LOWER = np.array([-2*math.pi, -2*math.pi, -math.pi, -2*math.pi, -2*math.pi, -2*math.pi])
ARM_UPPER = np.array([2*math.pi, 2*math.pi, math.pi, 2*math.pi, 2*math.pi, 2*math.pi])

SCHUNK_JOINT = 'schunk_left_finger_joint'
SCHUNK_MAX_OPEN = 0.033  # m

HOME_POSITION = np.array([0.0, -1.57, 0.0, -1.57, 0.0, 0.0])

# ─── UR5e DH Parameters ────────────────────────────────────────────────────────
# Standard Denavit-Hartenberg parameters (Universal Robots e-Series)
# Reference: UR5e Product Manual, section 6.1
_UR5E_DH = [
    # (a_i,     d_i,    alpha_i)
    (0.0,     0.1625,  math.pi / 2),   # joint 1
    (-0.4253, 0.0,     0.0),            # joint 2
    (-0.3922, 0.0,     0.0),            # joint 3
    (0.0,     0.1333,  math.pi / 2),   # joint 4
    (0.0,     0.0997, -math.pi / 2),   # joint 5
    (0.0,     0.0996,  0.0),            # joint 6
]

# Arm mount height on KAIROS base
# (Z distance between base_footprint and robot_arm_base_link)
ARM_MOUNT_Z = 0.84   # adjust if necessary


def _dh_matrix(a: float, d: float, alpha: float, theta: float) -> np.ndarray:
    """4x4 Denavit-Hartenberg homogeneous transformation matrix."""
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([
        [ct,  -st * ca,  st * sa, a * ct],
        [st,   ct * ca, -ct * sa, a * st],
        [0.0,       sa,       ca,      d],
        [0.0,      0.0,      0.0,    1.0],
    ], dtype=np.float64)


_T_BASE_TO_INERTIA = np.array([
    [-1.0,  0.0, 0.0, 0.0],
    [ 0.0, -1.0, 0.0, 0.0],
    [ 0.0,  0.0, 1.0, 0.0],
    [ 0.0,  0.0, 0.0, 1.0],
], dtype=np.float64)


def _ur5e_fk(q: np.ndarray) -> np.ndarray:
    """
    Complete UR5e forward kinematics via DH matrices.
    Returns TCP position (x, y, z) in robot_arm_base_link frame (REP-103).
    """
    T = _T_BASE_TO_INERTIA.copy()
    for i, (a, d, alpha) in enumerate(_UR5E_DH):
        T = T @ _dh_matrix(a, d, alpha, q[i])
    return T[:3, 3]


class _RosInterface(Node):
    """Internal ROS 2 node managing publishers and subscribers."""

    def __init__(self, namespace: str = '/robot'):
        super().__init__('kairos_rl_env', parameter_overrides=[
            Parameter('use_sim_time', Parameter.Type.BOOL, True)
        ])
        self.ns = namespace

        # ── State storage ──
        self.arm_positions = np.zeros(6)
        self.arm_velocities = np.zeros(6)
        self.gripper_position = 0.0
        self.base_pos = np.zeros(3)   # x, y, yaw
        self.base_vel = np.zeros(3)   # vx, vy, wz
        self.lidar_ranges = np.array([])
        # Real cube pose read from Gazebo
        self.cube_pos_world = np.array([2.5, 0.0, 0.82])
        self._lock = threading.Lock()

        # ── Subscribers ──
        self.create_subscription(
            JointState,
            f'{namespace}/joint_states',
            self._joint_state_cb, 10
        )
        self.create_subscription(
            LaserScan,
            f'{namespace}/front_laser/scan',
            self._laser_cb, 10
        )
        # Real odom topic = /robot/odom (remapped via rbkairos_control.urdf.xacro)
        self.create_subscription(
            Odometry,
            f'{namespace}/odom',
            self._odom_cb, 10
        )
        # Real cube pose via bridge gz->ROS
        self.create_subscription(
            Pose,
            '/model/rl_cube/pose_ros',
            self._cube_pose_cb, 10
        )

        # ── Publishers ──
        qos_best_effort = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        # Official mecanum_drive_controller topic in ROS 2 Jazzy (TwistStamped)
        self.cmd_vel_stamped_pub = self.create_publisher(
            TwistStamped, f'{namespace}/robotnik_base_control/reference', qos_best_effort
        )
        # Official unstamped topic if use_stamped_vel: false
        self.cmd_vel_ref_unstamped_pub = self.create_publisher(
            Twist, f'{namespace}/robotnik_base_control/reference_unstamped', qos_best_effort
        )
        # Fallback topics
        self.cmd_vel_pub = self.create_publisher(
            Twist, f'{namespace}/robotnik_base_control/cmd_vel_unstamped', 10
        )
        self.cmd_vel_fallback_pub = self.create_publisher(
            Twist, f'{namespace}/robotnik_base_control/cmd_vel', 10
        )
        self.arm_traj_pub = self.create_publisher(
            JointTrajectory,
            f'{namespace}/joint_trajectory_controller/joint_trajectory', 10
        )
        self.gripper_pub = self.create_publisher(
            JointTrajectory,
            f'{namespace}/schunk_egk50_controller/joint_trajectory', 10
        )
        self.marker_pub = self.create_publisher(
            MarkerArray, f'{namespace}/rl_collision_zones', 10
        )

    def publish_collision_zones(self, base_pos: np.ndarray, table_z: float):
        ma = MarkerArray()
        
        # 1. Table collision zone (world frame -> 'robot_odom' in RViz)
        m_table = Marker()
        m_table.header.frame_id = "robot_odom"
        m_table.header.stamp = self.get_clock().now().to_msg()
        m_table.ns = "rl_zones"
        m_table.id = 0
        m_table.type = Marker.CUBE
        m_table.action = Marker.ADD
        m_table.pose.position.x = 2.5
        m_table.pose.position.y = 0.0
        z_max = table_z - 0.04
        m_table.pose.position.z = z_max / 2.0
        m_table.pose.orientation.w = 1.0
        m_table.scale.x = 2.0
        m_table.scale.y = 2.0
        m_table.scale.z = z_max
        m_table.color.r = 1.0
        m_table.color.a = 0.3
        
        # 2. Self collision zone (base frame -> 'robot_base_link')
        m_base = Marker()
        m_base.header.frame_id = "robot_base_link"
        m_base.header.stamp = self.get_clock().now().to_msg()
        m_base.ns = "rl_zones"
        m_base.id = 1
        m_base.type = Marker.CUBE
        m_base.action = Marker.ADD
        m_base.pose.position.x = 0.0
        m_base.pose.position.y = 0.0
        m_base.pose.position.z = 0.4
        m_base.pose.orientation.w = 1.0
        m_base.scale.x = 1.1
        m_base.scale.y = 0.7
        m_base.scale.z = 0.8
        m_base.color.r = 1.0
        m_base.color.g = 0.0
        m_base.color.b = 0.0
        m_base.color.a = 0.3
        
        ma.markers = [m_table, m_base]
        self.marker_pub.publish(ma)

    def _joint_state_cb(self, msg: JointState):
        with self._lock:
            for i, name in enumerate(msg.name):
                if name in ARM_JOINTS:
                    idx = ARM_JOINTS.index(name)
                    if i < len(msg.position):
                        self.arm_positions[idx] = msg.position[i]
                    if i < len(msg.velocity):
                        self.arm_velocities[idx] = msg.velocity[i]
                elif name == SCHUNK_JOINT:
                    if i < len(msg.position):
                        self.gripper_position = msg.position[i]

    def _laser_cb(self, msg: LaserScan):
        with self._lock:
            self.lidar_ranges = np.array(msg.ranges, dtype=np.float32)

    def _odom_cb(self, msg: Odometry):
        with self._lock:
            self.base_pos[0] = msg.pose.pose.position.x
            self.base_pos[1] = msg.pose.pose.position.y
            q = msg.pose.pose.orientation
            siny_cosp = 2 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
            self.base_pos[2] = math.atan2(siny_cosp, cosy_cosp)

            self.base_vel[0] = msg.twist.twist.linear.x
            self.base_vel[1] = msg.twist.twist.linear.y
            self.base_vel[2] = msg.twist.twist.angular.z

    def _cube_pose_cb(self, msg: Pose):
        """Update real cube position from Gazebo."""
        with self._lock:
            self.cube_pos_world[0] = msg.position.x
            self.cube_pos_world[1] = msg.position.y
            self.cube_pos_world[2] = msg.position.z

    def get_state(self):
        with self._lock:
            return (
                self.arm_positions.copy(),
                self.arm_velocities.copy(),
                self.gripper_position,
                self.base_pos.copy(),
                self.base_vel.copy(),
                self.lidar_ranges.copy(),
                self.cube_pos_world.copy(),
            )

    def publish_cmd_vel(self, vx: float, vy: float, wz: float):
        msg = Twist()
        msg.linear.x = float(vx)
        msg.linear.y = float(vy)
        msg.angular.z = float(wz)

        stamped = TwistStamped()
        stamped.header.stamp = self.get_clock().now().to_msg()
        stamped.header.frame_id = 'robot_base_footprint'
        stamped.twist = msg

        # Publish on all available mobile base command channels
        self.cmd_vel_stamped_pub.publish(stamped)
        self.cmd_vel_ref_unstamped_pub.publish(msg)
        self.cmd_vel_pub.publish(msg)
        self.cmd_vel_fallback_pub.publish(msg)

    def publish_arm_target(self, target_positions: np.ndarray, duration_sec: float = 0.1):
        msg = JointTrajectory()
        msg.header.stamp.sec = 0
        msg.header.stamp.nanosec = 0
        msg.joint_names = ARM_JOINTS
        point = JointTrajectoryPoint()
        point.positions = [float(p) for p in target_positions]
        sec = int(duration_sec)
        nanosec = int((duration_sec - sec) * 1e9)
        point.time_from_start = Duration(sec=sec, nanosec=nanosec)
        msg.points = [point]
        self.arm_traj_pub.publish(msg)

    def publish_gripper(self, opening: float):
        """Send position command to Schunk gripper via JointTrajectory."""
        msg = JointTrajectory()
        msg.header.stamp.sec = 0
        msg.header.stamp.nanosec = 0
        msg.joint_names = ['schunk_left_finger_joint', 'schunk_right_finger_joint']
        point = JointTrajectoryPoint()
        point.positions = [float(opening), float(opening)]
        point.time_from_start = Duration(sec=0, nanosec=100_000_000)  # 100ms
        msg.points = [point]
        self.gripper_pub.publish(msg)

    def stop_base(self):
        self.publish_cmd_vel(0.0, 0.0, 0.0)


class KairosPickPlaceEnv(gym.Env):
    """
    Gymnasium environment for Pick & Place with the KAIROS robot.

    Action space (10D, continuous [-1, 1]):
        [0:3]  - Base velocities (vx, vy, wz)
        [3:9]  - Arm delta joint positions (Δq1..Δq6)
        [9]    - Gripper command (-1=close, +1=open)

    Observation space (39D):
        [0:6]   - Arm joint positions
        [6:12]  - Arm joint velocities
        [12]    - Gripper opening
        [13:16] - Base velocity (vx, vy, wz)
        [16:19] - TCP → Cube vector  (Full UR5e DH FK)
        [19:22] - Cube → Drop zone vector
        [22]    - Grasp status (cube physically lifted via Gazebo)
        [23:39] - LiDAR subsampled (16 rays)
    """

    metadata = {'render_modes': []}

    def __init__(self, config: dict = None):
        super().__init__()

        # ── Config ──
        if config is None:
            config = {}
        env_cfg = config.get('environment', {})
        self.control_hz = env_cfg.get('control_hz', 10)
        self.max_steps = env_cfg.get('max_steps', 400)

        arm_cfg = config.get('arm', {})
        self.delta_q_max = arm_cfg.get('delta_q_max', 0.05)
        self.home_position = np.array(
            arm_cfg.get('home_position', HOME_POSITION.tolist())
        )

        base_cfg = config.get('base', {})
        self.max_lin_vel = base_cfg.get('max_linear_vel', 1.0)
        self.max_ang_vel = base_cfg.get('max_angular_vel', 1.5)

        cube_cfg = config.get('cube', {})
        self.cube_size_range = cube_cfg.get('size_range', [0.03, 0.055])
        self.cube_mass_values = cube_cfg.get(
            'mass_values', [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        )
        self.spawn_x_range = cube_cfg.get('spawn_x_range', [2.0, 3.0])
        self.spawn_y_range = cube_cfg.get('spawn_y_range', [-0.5, 0.5])
        self.table_z = cube_cfg.get('table_z', 0.8)

        drop_cfg = config.get('drop_zone', {})
        self.drop_position = np.array(
            drop_cfg.get('position', [6.0, 0.0, 0.85])
        )

        reward_cfg = config.get('reward', {})
        self.approach_coeff = reward_cfg.get('approach_coeff', 1.0)
        self.grasp_bonus = reward_cfg.get('grasp_bonus', 10.0)
        self.transport_coeff = reward_cfg.get('transport_coeff', 1.0)
        self.drop_success_bonus = reward_cfg.get('drop_success_bonus', 50.0)
        self.speed_coeff = reward_cfg.get('speed_coeff', 0.1)
        self.collision_penalty = reward_cfg.get('collision_penalty', -20.0)
        self.drop_fail_penalty = reward_cfg.get('drop_fail_penalty', -30.0)
        self.action_reg_coeff = reward_cfg.get('action_reg_coeff', 0.01)
        self.heavy_bonus_coeff = reward_cfg.get('heavy_bonus_coeff', 2.0)

        lidar_cfg = config.get('lidar', {})
        self.num_lidar_samples = lidar_cfg.get('num_samples', 16)
        self.lidar_max_range = lidar_cfg.get('max_range', 10.0)
        self.collision_threshold = lidar_cfg.get('collision_threshold', 0.3)

        # ── Spaces ──
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(10,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(39,), dtype=np.float32
        )

        # ── ROS 2 ──
        if not rclpy.ok():
            rclpy.init()
        self._ros = _RosInterface()
        self._executor = MultiThreadedExecutor(num_threads=2)
        self._executor.add_node(self._ros)
        self._spin_thread = threading.Thread(
            target=self._executor.spin, daemon=True
        )
        self._spin_thread.start()

        # ── Episode state ──
        self._step_count = 0
        self._cube_name = 'rl_cube'
        self._cube_mass = 1.0
        self._cube_size = 0.04
        self._cube_spawn_pos = np.array([2.5, 0.0, 0.82])
        self._grasped = False
        self._grasped_prev = False
        self._cube_spawned = False   # Single spawn flag
        self._prev_tcp_cube_dist = None
        self._prev_cube_drop_dist = None

        # ── Collision counters (reset per episode) ──
        self._ep_table_collisions = 0
        self._ep_self_collisions = 0
        self._ep_cube_falls = 0

        # Wait for ROS 2 connections to settle (spin-based, no blocking sleep)
        self._spin_wait(1.5)

    # ─── Full Forward Kinematics via DH ────────────────────────────────────────

    def _get_tcp_world_position(self, arm_pos: np.ndarray,
                                 base_pos: np.ndarray) -> np.ndarray:
        """
        Compute 3D TCP position in World Frame.

        Uses full UR5e FK via Denavit-Hartenberg matrices (6 joints),
        then transforms the result to world frame via mobile base pose.
        """
        # 1. Full FK in robot_arm_base_link frame
        tcp_arm = _ur5e_fk(arm_pos)  # (x, y, z) in arm base frame

        # 2. Transform into world frame via mobile base pose
        base_x, base_y, base_yaw = base_pos[0], base_pos[1], base_pos[2]
        cos_yaw = math.cos(base_yaw)
        sin_yaw = math.sin(base_yaw)

        world_x = base_x + cos_yaw * tcp_arm[0] - sin_yaw * tcp_arm[1]
        world_y = base_y + sin_yaw * tcp_arm[0] + cos_yaw * tcp_arm[1]
        world_z = ARM_MOUNT_Z + tcp_arm[2]

        return np.array([world_x, world_y, world_z], dtype=np.float32)

    # ─── Spin-based wait (no blocking sleep) ────────────────────────────────

    def _spin_wait(self, duration_sec: float):
        """Wait for `duration_sec` while executing ROS callbacks."""
        end = self._ros.get_clock().now() + rclpy.duration.Duration(seconds=duration_sec)
        while rclpy.ok() and self._ros.get_clock().now() < end:
            rclpy.spin_once(self._ros, timeout_sec=0.01)

    # ─── Reset ────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 1. Teleport base to origin and stop
        gz_utils.set_model_pose('robot', 0.0, 0.0, 0.0, yaw=0.0)
        self._ros.stop_base()

        # 2. Arm to home position and open gripper (fast)
        self._ros.publish_arm_target(self.home_position, duration_sec=0.1)
        self._ros.publish_gripper(SCHUNK_MAX_OPEN)
        self._spin_wait(0.1)  # short wait instead of time.sleep

        # 3. Randomize cube parameters
        self._cube_size = random.uniform(*self.cube_size_range)
        self._cube_mass = random.choice(self.cube_mass_values)
        spawn_x = random.uniform(*self.spawn_x_range)
        spawn_y = random.uniform(*self.spawn_y_range)
        spawn_z = self.table_z + self._cube_size / 2.0 + 0.01
        self._cube_spawn_pos = np.array([spawn_x, spawn_y, spawn_z])

        # Spawn once, then teleport via set_pose
        if not self._cube_spawned:
            gz_utils.spawn_cube(
                self._cube_name, spawn_x, spawn_y, spawn_z,
                self._cube_size, self._cube_mass
            )
            self._cube_spawned = True
            self._spin_wait(0.5)
        else:
            gz_utils.set_model_pose(self._cube_name, spawn_x, spawn_y, spawn_z)
            self._spin_wait(0.1)  # near-instant teleportation

        # Initialize cube_pos_world to spawn position (updated by ROS)
        with self._ros._lock:
            self._ros.cube_pos_world = self._cube_spawn_pos.copy()

        # 4. Reset episode state
        self._step_count = 0
        self._grasped = False
        self._grasped_prev = False
        self._prev_tcp_cube_dist = None
        self._prev_cube_drop_dist = None
        self._ep_table_collisions = 0
        self._ep_self_collisions = 0
        self._ep_cube_falls = 0

        obs = self._get_observation()
        info = {'cube_mass': self._cube_mass, 'cube_size': self._cube_size}
        print(f"\n[RL Env] Episode Reset | Cube at ({spawn_x:.2f}, {spawn_y:.2f}, {spawn_z:.2f}) | Mass: {self._cube_mass}kg | Size: {self._cube_size*100:.1f}cm")
        return obs, info

    # ─── Step ─────────────────────────────────────────────────────────────────

    def step(self, action: np.ndarray):
        self._step_count += 1
        action = np.clip(action, -1.0, 1.0)

        # ── De-normalize and publish actions ──

        # Base velocity
        vx = float(action[0] * self.max_lin_vel)
        vy = float(action[1] * self.max_lin_vel)
        wz = float(action[2] * self.max_ang_vel)
        self._ros.publish_cmd_vel(vx, vy, wz)

        # Arm: incremental position
        arm_pos, _, _, _, _, _, _ = self._ros.get_state()
        delta_q = action[3:9] * self.delta_q_max
        target = np.clip(arm_pos + delta_q, ARM_LOWER, ARM_UPPER)
        self._ros.publish_arm_target(target, duration_sec=1.0 / self.control_hz)

        # Gripper: map [-1, 1] → [0, SCHUNK_MAX_OPEN]
        gripper_cmd = float((action[9] + 1.0) / 2.0 * SCHUNK_MAX_OPEN)
        self._ros.publish_gripper(gripper_cmd)

        # Wait for physics step (spin-based)
        self._spin_wait(1.0 / self.control_hz)

        # ── Grasp state based on real cube pose ──
        arm_pos, _, grip_pos, base_pos, _, _, cube_pos_real = self._ros.get_state()
        tcp_world = self._get_tcp_world_position(arm_pos, base_pos)

        # Grasp confirmed only if cube is physically lifted
        if not self._grasped:
            dist_to_cube = np.linalg.norm(tcp_world - cube_pos_real)
            cube_is_lifted = cube_pos_real[2] > (self.table_z + 0.05)
            if dist_to_cube < 0.08 and gripper_cmd < 0.015 and cube_is_lifted:
                self._grasped = True

        if self._grasped:
            # Release if gripper opens
            if gripper_cmd > 0.02:
                self._grasped = False

        # ── Get observation ──
        obs = self._get_observation()

        # ── Compute reward ──
        reward, terminated, info = self._compute_reward(action, obs, cube_pos_real, tcp_world, base_pos)

        # ── Publish Visualization ──
        self._ros.publish_collision_zones(base_pos, self.table_z)

        # ── Truncation ──
        truncated = self._step_count >= self.max_steps

        if terminated or truncated:
            self._ros.stop_base()
            outcome = info.get('success', False)
            print(f"\n{'='*60}")
            print(f"[RL Env] === EPISODE COMPLETED (Step {self._step_count}/{self.max_steps}) ===")
            print(f"  Result            : {'✅ SUCCESS' if outcome else '❌ Failed'}")
            print(f"  Table Collisions  : {self._ep_table_collisions}")
            print(f"  Self-Collisions   : {self._ep_self_collisions}")
            print(f"  Cube Drops        : {self._ep_cube_falls}")
            print(f"  Cube Grasped      : {self._grasped}")
            print(f"{'='*60}\n")
            info['ep_table_collisions'] = self._ep_table_collisions
            info['ep_self_collisions'] = self._ep_self_collisions
            info['ep_cube_falls'] = self._ep_cube_falls

        # ── Step Logging (with FK & cube pose audit) ──
        if self._step_count % 10 == 0 or terminated or truncated:
            tcp_to_cube_dist = float(np.linalg.norm(obs[16:19]))
            # Audit FK: TCP position in world frame
            tcp_world_audit = self._get_tcp_world_position(arm_pos, base_pos)
            print(f"[RL Env] Step {self._step_count:03d}/{self.max_steps} | "
                  f"Base=({vx:+.2f}, {vy:+.2f}, {wz:+.2f}) | "
                  f"TCP_world=({tcp_world_audit[0]:.2f},{tcp_world_audit[1]:.2f},{tcp_world_audit[2]:.2f}) | "
                  f"CubeReal=({cube_pos_real[0]:.2f},{cube_pos_real[1]:.2f},{cube_pos_real[2]:.2f}) | "
                  f"Dist: {tcp_to_cube_dist:.2f}m | "
                  f"R={reward:+.2f} | Grasped: {self._grasped}")

        return obs, reward, terminated, truncated, info

    # ─── Observation ──────────────────────────────────────────────────────────

    def _get_observation(self) -> np.ndarray:
        arm_pos, arm_vel, grip_pos, base_pos, base_vel, lidar, cube_pos_real = \
            self._ros.get_state()
        tcp_world = self._get_tcp_world_position(arm_pos, base_pos)

        # 3D World Vectors
        tcp_to_cube_w = (cube_pos_real - tcp_world).astype(np.float32)
        cube_to_drop_w = (self.drop_position - cube_pos_real).astype(np.float32)
        cube_lifted = 1.0 if self._grasped else 0.0

        # Projection into mobile base local coordinate system (Body Frame)
        # +X_robot = in front of robot, +Y_robot = to the left of robot
        # Thus action vx > 0 moves robot directly toward a cube at +X_robot
        base_yaw = base_pos[2]
        cos_yaw, sin_yaw = math.cos(base_yaw), math.sin(base_yaw)

        tcp_to_cube = np.array([
            cos_yaw * tcp_to_cube_w[0] + sin_yaw * tcp_to_cube_w[1],
            -sin_yaw * tcp_to_cube_w[0] + cos_yaw * tcp_to_cube_w[1],
            tcp_to_cube_w[2],
        ], dtype=np.float32)

        cube_to_drop = np.array([
            cos_yaw * cube_to_drop_w[0] + sin_yaw * cube_to_drop_w[1],
            -sin_yaw * cube_to_drop_w[0] + cos_yaw * cube_to_drop_w[1],
            cube_to_drop_w[2],
        ], dtype=np.float32)

        # LiDAR subsampling
        if len(lidar) > 0:
            indices = np.linspace(0, len(lidar) - 1, self.num_lidar_samples, dtype=int)
            lidar_sub = lidar[indices]
            lidar_sub = np.where(np.isfinite(lidar_sub), lidar_sub, self.lidar_max_range)
        else:
            lidar_sub = np.full(self.num_lidar_samples, self.lidar_max_range, dtype=np.float32)

        obs = np.concatenate([
            arm_pos.astype(np.float32),              # 6
            arm_vel.astype(np.float32),              # 6
            np.array([grip_pos], dtype=np.float32),  # 1
            base_vel.astype(np.float32),             # 3
            tcp_to_cube,                             # 3 (robot local frame)
            cube_to_drop,                            # 3 (robot local frame)
            np.array([cube_lifted], dtype=np.float32),  # 1
            lidar_sub.astype(np.float32),            # 16
        ]).astype(np.float32)

        return obs

    # ─── Reward ───────────────────────────────────────────────────────────────

    def _compute_reward(self, action: np.ndarray, obs: np.ndarray,
                        cube_pos_real: np.ndarray, tcp_world: np.ndarray, base_pos: np.ndarray):
        info = {}
        terminated = False

        tcp_to_cube = obs[16:19]
        cube_to_drop = obs[19:22]
        cube_lifted = obs[22]
        lidar_sub = obs[23:39]

        tcp_cube_dist = float(np.linalg.norm(tcp_to_cube))
        cube_drop_dist = float(np.linalg.norm(cube_to_drop))

        # ── Reward components (isolated for diagnostics) ──
        r_approach = 0.0
        r_approach_delta = 0.0
        r_grasp_bonus = 0.0
        r_transport = 0.0
        r_transport_delta = 0.0
        r_drop_success = 0.0
        r_cube_fell = 0.0
        r_collision = 0.0
        r_action_reg = 0.0

        # ── Phase 1: Approach reward ──
        if not self._grasped:
            if self._prev_tcp_cube_dist is not None:
                delta = self._prev_tcp_cube_dist - tcp_cube_dist
                r_approach_delta = delta * 100.0

        # ── Phase 2: Grasp bonus (applied once upon grasp) ──
        if self._grasped and not self._grasped_prev:
            mass_factor = self._cube_mass / max(self.cube_mass_values)
            r_grasp_bonus = self.grasp_bonus + self.grasp_bonus * mass_factor * self.heavy_bonus_coeff
            info['grasped'] = True

        # ── Phase 3: Transport reward ──
        if self._grasped:
            r_transport = -self.transport_coeff * cube_drop_dist
            if self._prev_cube_drop_dist is not None:
                delta = self._prev_cube_drop_dist - cube_drop_dist
                r_transport_delta = delta * 5.0

        # ── Phase 4: Drop success (based on real cube pose) ──
        cube_above_drop = (
            cube_drop_dist < 0.20
            and not self._grasped
            and cube_pos_real[2] > self.table_z
        )
        if cube_above_drop:
            time_remaining = self.max_steps - self._step_count
            mass_factor = self._cube_mass / max(self.cube_mass_values)
            r_drop_success = (self.drop_success_bonus
                              + time_remaining * self.speed_coeff
                              + self.drop_success_bonus * mass_factor * self.heavy_bonus_coeff)
            terminated = True
            info['success'] = True

        # ── Penalty: cube dropped on the floor ──
        if cube_pos_real[2] < 0.05:
            r_cube_fell = self.drop_fail_penalty
            self._ep_cube_falls += 1
            terminated = True
            info['cube_dropped'] = True

        # ── Penalty: collision (Table collision zone) ──
        # Table is approximately between X=1.5 and 3.5, Y=-1.0 and 1.0. Z table = self.table_z (0.8)
        if tcp_world[2] < (self.table_z - 0.04) and 1.5 < tcp_world[0] < 3.5 and -1.0 < tcp_world[1] < 1.0:
            r_collision = self.collision_penalty
            self._ep_table_collisions += 1
            info['collision'] = True
            info['crash_table'] = True

        # ── Penalty: Self-collision (Base) — 2x penalty ──
        base_yaw = base_pos[2]
        cos_yaw, sin_yaw = math.cos(-base_yaw), math.sin(-base_yaw)
        dx = tcp_world[0] - base_pos[0]
        dy = tcp_world[1] - base_pos[1]
        tcp_base_x = cos_yaw * dx - sin_yaw * dy
        tcp_base_y = sin_yaw * dx + cos_yaw * dy
        tcp_base_z = tcp_world[2]

        if (-0.55 < tcp_base_x < 0.55) and (-0.35 < tcp_base_y < 0.35) and (tcp_base_z < 0.8):
            r_collision = self.collision_penalty * 2.0  # 2x vs table collision
            self._ep_self_collisions += 1
            info['collision'] = True
            info['self_collision'] = True

        # ── Action regularization ──
        r_action_reg = -float(self.action_reg_coeff * np.sum(action ** 2))

        # ── Total ──
        reward = (r_approach + r_approach_delta
                  + r_grasp_bonus
                  + r_transport + r_transport_delta
                  + r_drop_success
                  + r_cube_fell
                  + r_collision
                  + r_action_reg)

        # ── Reward decomposition log (every 10 steps) ──
        if self._step_count % 10 == 0:
            print(f"  [REWARD] Approach={r_approach:+.2f} Δappr={r_approach_delta:+.3f} | "
                  f"Grasp={r_grasp_bonus:+.1f} | "
                  f"Transport={r_transport:+.2f} Δtrans={r_transport_delta:+.3f} | "
                  f"DropOK={r_drop_success:+.1f} | "
                  f"Fell={r_cube_fell:+.1f} Collision={r_collision:+.1f} | "
                  f"ActReg={r_action_reg:+.3f} | "
                  f"TOTAL={reward:+.2f}")

        # Store decomposition in info for TensorBoard
        info['r_approach'] = r_approach + r_approach_delta
        info['r_grasp'] = r_grasp_bonus
        info['r_transport'] = r_transport + r_transport_delta
        info['r_drop'] = r_drop_success
        info['r_collision'] = r_collision
        info['r_cube_fell'] = r_cube_fell
        info['r_action_reg'] = r_action_reg

        # Update previous distances and grasp state
        self._prev_tcp_cube_dist = tcp_cube_dist
        self._prev_cube_drop_dist = cube_drop_dist
        self._grasped_prev = self._grasped

        return reward, terminated, info

    def close(self):
        """Cleanup."""
        try:
            if rclpy.ok():
                self._ros.stop_base()
        except Exception:
            pass
        gz_utils.delete_model(self._cube_name)
        try:
            self._executor.shutdown()
            if rclpy.ok():
                self._ros.destroy_node()
        except Exception:
            pass
