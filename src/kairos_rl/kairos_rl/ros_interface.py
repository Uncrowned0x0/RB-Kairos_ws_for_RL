"""
Shared ROS 2 node for KAIROS Curriculum Learning.
Manages publishers and subscribers for UR5e arm and Schunk EGK50 gripper.
"""
import threading
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from visualization_msgs.msg import Marker, MarkerArray


# ─── Joint names ───────────────────────────────────────────────────────────────
ARM_JOINTS = [
    'robot_arm_shoulder_pan_joint',
    'robot_arm_shoulder_lift_joint',
    'robot_arm_elbow_joint',
    'robot_arm_wrist_1_joint',
    'robot_arm_wrist_2_joint',
    'robot_arm_wrist_3_joint',
]

SCHUNK_JOINT = 'schunk_left_finger_joint'
SCHUNK_MAX_OPEN = 0.033  # m


class ArmRosInterface(Node):
    """
    ROS 2 node dedicated to UR5e arm + Schunk gripper control.
    Used by all Curriculum Learning levels.
    """

    def __init__(self, namespace: str = '/robot'):
        super().__init__('kairos_curriculum_env', parameter_overrides=[
            Parameter('use_sim_time', Parameter.Type.BOOL, True)
        ])
        self.ns = namespace

        # ── State storage ──
        self.arm_positions = np.zeros(6)
        self.arm_velocities = np.zeros(6)
        self.gripper_position = 0.0
        # Ground-truth cube world pose from Gazebo
        self.cube_pos_world = np.array([0.4, 0.0, 0.78])
        self._lock = threading.Lock()

        # ── Subscribers ──
        # joint_state_broadcaster publishes with RELIABLE QoS. In DDS, a BEST_EFFORT
        # subscriber cannot match a RELIABLE publisher. Using RELIABLE with depth=10.
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(
            JointState,
            f'{namespace}/joint_states',
            self._joint_state_cb, qos
        )
        # Real cube pose via gz->ROS bridge
        self.create_subscription(
            Pose,
            '/model/rl_cube/pose_ros',
            self._cube_pose_cb, 10
        )

        # ── Publishers ──
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

    # ─── Callbacks ────────────────────────────────────────────────────────────

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

    def _cube_pose_cb(self, msg: Pose):
        with self._lock:
            self.cube_pos_world[0] = msg.position.x
            self.cube_pos_world[1] = msg.position.y
            self.cube_pos_world[2] = msg.position.z

    # ─── State getter ─────────────────────────────────────────────────────────

    def get_state(self):
        """Return current state of arm, gripper, and cube."""
        with self._lock:
            return (
                self.arm_positions.copy(),
                self.arm_velocities.copy(),
                self.gripper_position,
                self.cube_pos_world.copy(),
            )

    # ─── Publishers ───────────────────────────────────────────────────────────

    def publish_arm_target(self, positions: np.ndarray, duration_sec: float = 0.1):
        """Publish joint position command for UR5e arm."""
        msg = JointTrajectory()
        msg.joint_names = ARM_JOINTS
        point = JointTrajectoryPoint()
        point.positions = positions.tolist()
        nsec = int(duration_sec * 1e9)
        point.time_from_start = Duration(sec=int(nsec // 1_000_000_000),
                                         nanosec=int(nsec % 1_000_000_000))
        msg.points = [point]
        self.arm_traj_pub.publish(msg)

    def publish_gripper(self, position: float):
        """Publish position command for Schunk EGK50 gripper."""
        msg = JointTrajectory()
        msg.joint_names = ['schunk_left_finger_joint', 'schunk_right_finger_joint']
        point = JointTrajectoryPoint()
        point.positions = [position, position]
        point.time_from_start = Duration(sec=0, nanosec=100_000_000)
        msg.points = [point]
        self.gripper_pub.publish(msg)

    def publish_collision_zones(self):
        """Publish collision zones for RViz (KAIROS chassis)."""
        ma = MarkerArray()

        m_base = Marker()
        m_base.header.frame_id = "robot_base_link"
        m_base.header.stamp = self.get_clock().now().to_msg()
        m_base.ns = "rl_zones"
        m_base.id = 0
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

        ma.markers = [m_base]
        self.marker_pub.publish(ma)
