#!/usr/bin/env python3
# ==============================================================================
# Author: Kamil BENMADI
# Email: kamil.benmadi@sigma-clermont.fr
# GitHub: https://github.com/Uncrowned0x0
# ==============================================================================
"""
kairos_robot.launch.py — Spawn RB-KAIROS robot and controllers (Terminal 2).

Spawns:
  1. RB-KAIROS mobile base + UR5e arm + selected gripper (schunk_egk50 or tesollo_dg5f).
  2. ros2_control hardware interface and controllers (joint_state_broadcaster,
     robotnik_base_control, joint_trajectory_controller, gripper controller).
  3. Sensor bridges and tactile mock nodes (if tesollo_dg5f).
  4. RViz2 visualization (optional, run_rviz:=true by default).

Usage:
  ros2 launch kairos_bringup kairos_robot.launch.py
  ros2 launch kairos_bringup kairos_robot.launch.py gripper_type:=schunk_egk50
  ros2 launch kairos_bringup kairos_robot.launch.py gripper_type:=tesollo_dg5f run_rviz:=false
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    declared_args = [
        DeclareLaunchArgument(
            "robot_id",
            default_value="robot",
            description="Unique robot identifier (ROS 2 namespace)",
        ),
        DeclareLaunchArgument(
            "gripper_type",
            default_value="schunk_egk50",
            description="Supported gripper type: schunk_egk50 | tesollo_dg5f",
        ),
        DeclareLaunchArgument(
            "run_rviz",
            default_value="true",
            description="Launch RViz2 together with robot (true/false)",
        ),
        DeclareLaunchArgument(
            "rviz_config",
            default_value="",
            description="Path to custom RViz config file (empty for default)",
        ),
        DeclareLaunchArgument(
            "low_performance_simulation",
            default_value="false",
            description="Reduce sensor rendering overhead for lower CPU/GPU usage (true/false)",
        ),
        DeclareLaunchArgument(
            "x", default_value="0.0", description="Initial X spawn coordinate"
        ),
        DeclareLaunchArgument(
            "y", default_value="0.0", description="Initial Y spawn coordinate"
        ),
        DeclareLaunchArgument(
            "z", default_value="0.0", description="Initial Z spawn coordinate"
        ),
    ]

    robot_id = LaunchConfiguration("robot_id")
    gripper_type = LaunchConfiguration("gripper_type")
    run_rviz = LaunchConfiguration("run_rviz")
    rviz_config = LaunchConfiguration("rviz_config")
    low_perf = LaunchConfiguration("low_performance_simulation")
    x = LaunchConfiguration("x")
    y = LaunchConfiguration("y")
    z = LaunchConfiguration("z")

    spawn_robot_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("robotnik_gazebo_ignition"),
                "launch",
                "spawn_robot.launch.py",
            ])
        ),
        launch_arguments={
            "robot_id": robot_id,
            "robot": "rbkairos",
            "robot_model": "rbkairos_plus",
            "robot_xacro_path": PathJoinSubstitution([
                FindPackageShare("rbkairos_description"),
                "robots",
                "rbkairos_ur5_qbhand.urdf.xacro",
            ]),
            "arm_type": "ur5e",
            "gripper_type": gripper_type,
            "run_rviz": run_rviz,
            "rviz_config": rviz_config,
            "low_performance_simulation": low_perf,
            "x": x,
            "y": y,
            "z": z,
        }.items(),
    )

    return LaunchDescription(declared_args + [spawn_robot_launch])
