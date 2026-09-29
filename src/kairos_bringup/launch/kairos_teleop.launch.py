#!/usr/bin/env python3
# ==============================================================================
# Author: Kamil BENMADI
# Email: kamil.benmadi@sigma-clermont.fr
# GitHub: https://github.com/Uncrowned0x0
# ==============================================================================
"""
kairos_teleop.launch.py — RB-KAIROS keyboard teleoperation launch file

Launches the safe interactive teleoperation node kairos_teleop_keyboard which publishes:
- TwistStamped on /<robot_id>/robotnik_base_control/reference (for mecanum_drive_controller)
- Twist on /<robot_id>/robotnik_base_control/reference_unstamped (unstamped controller setups)
- Twist on /<robot_id>/cmd_vel

Usage:
  ros2 launch kairos_bringup kairos_teleop.launch.py
  ros2 launch kairos_bringup kairos_teleop.launch.py use_xterm:=true
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():

    robot_id_arg = DeclareLaunchArgument(
        "robot_id",
        default_value="robot",
        description="Unique robot identifier (ROS 2 namespace)"
    )
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use Gazebo simulation clock"
    )
    use_xterm_arg = DeclareLaunchArgument(
        "use_xterm",
        default_value="false",
        description="Open dedicated xterm window (true) or run directly in current terminal (false)"
    )

    robot_id = LaunchConfiguration("robot_id")
    use_sim_time = LaunchConfiguration("use_sim_time")
    use_xterm = LaunchConfiguration("use_xterm")

    # Frame ID for TwistStamped: robot_base_footprint (or base_footprint if namespace is empty)
    frame_id = PythonExpression(["'", robot_id, "_base_footprint' if '", robot_id, "' else 'base_footprint'"])

    teleop_params = {
        "use_sim_time": use_sim_time,
        "frame_id": frame_id,
        "publish_stamped": True,
        "publish_unstamped": True,
        "publish_reference_unstamped": True,
        "cmd_vel_topic": "cmd_vel",
        "stamped_cmd_vel_topic": "robotnik_base_control/reference",
        "unstamped_reference_topic": "robotnik_base_control/reference_unstamped",
        "deadman_timeout": 0.60,
    }

    # ── Teleop node in dedicated xterm window (if use_xterm:=true) ────────────
    teleop_node_xterm = Node(
        package="kairos_bringup",
        executable="kairos_teleop_keyboard.py",
        name="kairos_teleop_keyboard",
        namespace=robot_id,
        output="screen",
        prefix="xterm -e",
        parameters=[teleop_params],
        condition=IfCondition(use_xterm),
    )

    # ── Direct teleop node in current terminal (if use_xterm:=false) ──────────
    teleop_node_direct = Node(
        package="kairos_bringup",
        executable="kairos_teleop_keyboard.py",
        name="kairos_teleop_keyboard",
        namespace=robot_id,
        output="screen",
        emulate_tty=True,
        parameters=[teleop_params],
        condition=UnlessCondition(use_xterm),
    )

    return LaunchDescription([
        robot_id_arg,
        use_sim_time_arg,
        use_xterm_arg,
        teleop_node_xterm,
        teleop_node_direct,
    ])
