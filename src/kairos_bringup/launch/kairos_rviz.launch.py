#!/usr/bin/env python3
# ==============================================================================
# Author: Kamil BENMADI
# Email: kamil.benmadi@sigma-clermont.fr
# GitHub: https://github.com/Uncrowned0x0
# ==============================================================================
"""
kairos_rviz.launch.py — Standalone RViz2 launcher (Terminal 2 alternative or Terminal 4).

Launches:
  RViz2 configured with robot_odom fixed frame and standard RB-KAIROS RViz profile.

Usage:
  ros2 launch kairos_bringup kairos_rviz.launch.py
  ros2 launch kairos_bringup kairos_rviz.launch.py robot_id:=robot
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    declared_args = [
        DeclareLaunchArgument(
            "robot_id",
            default_value="robot",
            description="Unique robot identifier (ROS 2 namespace)",
        ),
        DeclareLaunchArgument(
            "rviz_config",
            default_value=PathJoinSubstitution([
                FindPackageShare("robotnik_gazebo_ignition"),
                "config",
                "rviz_config.rviz",
            ]),
            description="Path to RViz configuration file",
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Use Gazebo simulation clock (/clock)",
        ),
    ]

    robot_id = LaunchConfiguration("robot_id")
    rviz_config = LaunchConfiguration("rviz_config")
    use_sim_time = LaunchConfiguration("use_sim_time")

    fixed_frame = PythonExpression(["'", robot_id, "_odom' if '", robot_id, "' else 'odom'"])

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        namespace=robot_id,
        arguments=[
            "-f", fixed_frame,
            "-d", rviz_config,
            "-t", [robot_id, " - RB-KAIROS RViz2"],
        ],
        parameters=[{"use_sim_time": use_sim_time}],
        output="screen",
    )

    return LaunchDescription(declared_args + [rviz_node])
