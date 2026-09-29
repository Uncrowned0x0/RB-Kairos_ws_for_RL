#!/usr/bin/env python3
# ==============================================================================
# Author: Kamil BENMADI
# Email: kamil.benmadi@sigma-clermont.fr
# GitHub: https://github.com/Uncrowned0x0
# ==============================================================================
"""
kairos_nav2.launch.py — Nav2 autonomous navigation for Kairos (without Gazebo)

Launches only the Nav2 navigation stack. Useful after starting Gazebo simulation
via kairos_sim_complete.launch.py with launch_nav2:=False, or to restart Nav2.

Usage from inside container:
  ros2 launch kairos_bringup kairos_nav2.launch.py
  ros2 launch kairos_bringup kairos_nav2.launch.py robot_id:=robot

Available arguments:
  robot_id : Robot namespace (default: robot)
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    # ── Arguments ─────────────────────────────────────────────────────────────
    robot_id_arg = DeclareLaunchArgument(
        "robot_id",
        default_value="robot",
        description="ROS 2 robot namespace",
    )
    robot_id = LaunchConfiguration("robot_id")

    # ── Nav2 Configuration File ───────────────────────────────────────────────
    nav2_params_file = os.path.join(
        get_package_share_directory("rbkairos_description"),
        "config",
        "nav2_params.yaml",
    )

    # Topic remappings: Nav2 publishes cmd_vel to /robot/nav2_cmd_vel -> twist_relay
    # converts it into TwistStamped for the mecanum drive controller
    remappings = [
        ("cmd_vel", "/robot/nav2_cmd_vel"),
        ("/robot/cmd_vel", "/robot/nav2_cmd_vel"),
        ("odom", "/robot/odom"),
        ("goal_pose", "/goal_pose"),
        ("/robot/goal_pose", "/goal_pose"),
    ]

    # ── Twist -> TwistStamped Relay ───────────────────────────────────────────
    twist_relay_node = Node(
        package="rbkairos_description",
        executable="twist_relay.py",
        name="twist_relay_nav2",
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    # ── Lifecycle Manager ─────────────────────────────────────────────────────
    lifecycle_manager_node = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_navigation",
        output="screen",
        namespace="robot",
        parameters=[
            {"use_sim_time": True},
            {"autostart": True},
            {
                "node_names": [
                    "controller_server",
                    "planner_server",
                    "behavior_server",
                    "bt_navigator",
                    "waypoint_follower",
                ]
            },
        ],
    )

    # ── Controller Server (RPP — local trajectory tracking) ───────────────────
    controller_server_node = Node(
        package="nav2_controller",
        executable="controller_server",
        name="controller_server",
        output="screen",
        namespace="robot",
        parameters=[nav2_params_file],
        remappings=remappings,
    )

    # ── Planner Server (NavFn — global path planning) ─────────────────────────
    planner_server_node = Node(
        package="nav2_planner",
        executable="planner_server",
        name="planner_server",
        output="screen",
        namespace="robot",
        parameters=[nav2_params_file],
        remappings=remappings,
    )

    # ── Behavior Server (recovery behaviors) ──────────────────────────────────
    behavior_server_node = Node(
        package="nav2_behaviors",
        executable="behavior_server",
        name="behavior_server",
        output="screen",
        namespace="robot",
        parameters=[nav2_params_file],
        remappings=remappings,
    )

    # ── BT Navigator (behavior tree, listens to /goal_pose) ───────────────────
    bt_navigator_node = Node(
        package="nav2_bt_navigator",
        executable="bt_navigator",
        name="bt_navigator",
        output="screen",
        namespace="robot",
        parameters=[nav2_params_file],
        remappings=remappings,
    )

    # ── Waypoint Follower ─────────────────────────────────────────────────────
    waypoint_follower_node = Node(
        package="nav2_waypoint_follower",
        executable="waypoint_follower",
        name="waypoint_follower",
        output="screen",
        namespace="robot",
        parameters=[nav2_params_file],
        remappings=remappings,
    )

    return LaunchDescription([
        robot_id_arg,
        twist_relay_node,
        controller_server_node,
        planner_server_node,
        behavior_server_node,
        bt_navigator_node,
        waypoint_follower_node,
        lifecycle_manager_node,
    ])
