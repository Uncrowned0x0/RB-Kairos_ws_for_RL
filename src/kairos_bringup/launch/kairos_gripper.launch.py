#!/usr/bin/env python3
# ==============================================================================
# Author: Kamil BENMADI
# Email: kamil.benmadi@sigma-clermont.fr
# GitHub: https://github.com/Uncrowned0x0
# ==============================================================================
"""
kairos_gripper.launch.py — Control of Tesollo DG-5F-R gripper in Gazebo

Launches Robot State Publisher and ros2_control controllers for the Tesollo
DG-5F-R robotic hand (5 fingers, 20 joints).

Used AFTER starting the main simulation (kairos_sim_complete.launch.py),
if the gripper needs to be controlled independently.

Usage from inside container:
  ros2 launch kairos_bringup kairos_gripper.launch.py
  ros2 launch kairos_bringup kairos_gripper.launch.py hand:=right

Available arguments:
  hand : Which hand to launch (right | left | both)
         right -> right hand DG-5F-R (default)
         left  -> left hand DG-5F-L
         both  -> both hands
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import LaunchConfigurationEquals
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    # ── Arguments ─────────────────────────────────────────────────────────────
    hand_arg = DeclareLaunchArgument(
        "hand",
        default_value="right",
        description="Which hand to launch: right | left | both",
    )

    # ── Right Hand ────────────────────────────────────────────────────────────
    gripper_right = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("dg5f_gz"),
                "launch",
                "dg5f_right_gz.launch.py",
            ])
        ),
        condition=LaunchConfigurationEquals("hand", "right"),
    )

    # ── Left Hand ─────────────────────────────────────────────────────────────
    gripper_left = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("dg5f_gz"),
                "launch",
                "dg5f_left_gz.launch.py",
            ])
        ),
        condition=LaunchConfigurationEquals("hand", "left"),
    )

    # ── Both Hands ────────────────────────────────────────────────────────────
    gripper_both = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("dg5f_gz"),
                "launch",
                "dg5f_both_gz.launch.py",
            ])
        ),
        condition=LaunchConfigurationEquals("hand", "both"),
    )

    return LaunchDescription([
        hand_arg,
        gripper_right,
        gripper_left,
        gripper_both,
    ])
