#!/usr/bin/env python3
# ==============================================================================
# Author: Kamil BENMADI
# Email: kamil.benmadi@sigma-clermont.fr
# GitHub: https://github.com/Uncrowned0x0
# ==============================================================================
"""
kairos_sim_complete.launch.py — Complete Kairos simulation in a single launch file

Launches:
  1. Gazebo Harmonic with chosen world (default: labo)
  2. Kairos+ robot (omnidirectional base + UR5e arm) with gripper
     (Tesollo DG-5F-R / Schunk / QB Hand)
  3. Nav2 autonomous navigation

Usage inside container after compilation:
  ros2 launch kairos_bringup kairos_sim_complete.launch.py
  ros2 launch kairos_bringup kairos_sim_complete.launch.py gripper_type:=tesollo_dg5f world:=empty

Available arguments:
  world          : Gazebo world (empty | labo | demo | lightweight_scene)
  gripper_type   : Gripper type (schunk_egk50 | tesollo_dg5f)
  robot_id       : Robot namespace (default: robot)
  run_rviz       : Launch RViz (True | False)
  launch_nav2    : Launch Nav2 autonomous navigation (True | False)
  low_performance_simulation : Low-performance mode for modest hardware
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    # ── Arguments ─────────────────────────────────────────────────────────────
    declared_args = [
        DeclareLaunchArgument(
            "robot_id", default_value="robot",
            description="ROS 2 robot namespace"
        ),
        DeclareLaunchArgument(
            "world", default_value="labo",
            description="Gazebo world: empty | labo | demo | lightweight_scene"
        ),
        DeclareLaunchArgument(
            "gripper_type", default_value="schunk_egk50",
            description="Supported gripper type: schunk_egk50 | tesollo_dg5f"
        ),
        DeclareLaunchArgument(
            "run_rviz", default_value="True",
            description="Launch RViz (True/False)"
        ),
        DeclareLaunchArgument(
            "launch_nav2", default_value="True",
            description="Launch Nav2 autonomous navigation (True/False)"
        ),
        DeclareLaunchArgument(
            "low_performance_simulation", default_value="False",
            description="Low performance simulation mode for low-spec PCs"
        ),
    ]

    robot_id = LaunchConfiguration("robot_id")
    world = LaunchConfiguration("world")
    gripper_type = LaunchConfiguration("gripper_type")
    run_rviz = LaunchConfiguration("run_rviz")
    launch_nav2 = LaunchConfiguration("launch_nav2")
    low_perf = LaunchConfiguration("low_performance_simulation")

    # ── 1. Gazebo World ───────────────────────────────────────────────────────
    gazebo_world = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("robotnik_gazebo_ignition"),
                "launch/spawn_world.launch.py",
            ])
        ),
        launch_arguments={
            "robot_id": robot_id,
            "gui": "true",
            "world_path": PathJoinSubstitution([
                FindPackageShare("robotnik_gazebo_ignition"),
                "worlds/",
                [world, ".world"],
            ]),
        }.items(),
    )

    # ── 2. Robot (spawn + ros2_control controllers) ───────────────────────────
    gazebo_robot = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("robotnik_gazebo_ignition"),
                "launch/spawn_robot.launch.py",
            ])
        ),
        launch_arguments={
            "robot_id": robot_id,
            "robot": "rbkairos",
            "robot_model": "rbkairos_plus",
            "robot_xacro_path": PathJoinSubstitution([
                FindPackageShare("rbkairos_description"),
                "robots/rbkairos_ur5_qbhand.urdf.xacro",
            ]),
            "arm_type": "ur5e",
            "gripper_type": gripper_type,
            "run_rviz": run_rviz,
            "low_performance_simulation": low_perf,
        }.items(),
    )

    # ── 3. Nav2 Navigation (delayed by 20s to let Gazebo & robot stabilize) ───
    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("rbkairos_description"),
                "launch/nav2_full.launch.py",
            ])
        ),
        condition=IfCondition(launch_nav2),
    )
    delayed_nav2 = TimerAction(period=20.0, actions=[nav2_launch])

    return LaunchDescription(declared_args + [
        gazebo_world,
        gazebo_robot,
        delayed_nav2,
    ])
