#!/usr/bin/env python3
# ==============================================================================
# Author: Kamil BENMADI
# Email: kamil.benmadi@sigma-clermont.fr
# GitHub: https://github.com/Uncrowned0x0
# ==============================================================================
"""
kairos_world.launch.py — Launch Gazebo simulation world alone (Terminal 1).

Starts:
  1. Gazebo Harmonic server and GUI with the requested world (default: labo).
  2. ROS-Gazebo clock bridge (/clock) for simulation time synchronization.

Usage:
  ros2 launch kairos_bringup kairos_world.launch.py
  ros2 launch kairos_bringup kairos_world.launch.py world:=labo gui:=true
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    declared_args = [
        DeclareLaunchArgument(
            "world",
            default_value="labo",
            description="Gazebo world name: labo | empty | demo | lightweight_scene",
        ),
        DeclareLaunchArgument(
            "gui",
            default_value="true",
            description="Enable Gazebo GUI (true/false)",
        ),
        DeclareLaunchArgument(
            "robot_id",
            default_value="robot",
            description="Robot namespace identifier",
        ),
    ]

    world = LaunchConfiguration("world")
    gui = LaunchConfiguration("gui")
    robot_id = LaunchConfiguration("robot_id")

    spawn_world_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("robotnik_gazebo_ignition"),
                "launch",
                "spawn_world.launch.py",
            ])
        ),
        launch_arguments={
            "robot_id": robot_id,
            "gui": gui,
            "world": world,
            "world_path": PathJoinSubstitution([
                FindPackageShare("robotnik_gazebo_ignition"),
                "worlds",
                [world, ".world"],
            ]),
        }.items(),
    )

    return LaunchDescription(declared_args + [spawn_world_launch])
