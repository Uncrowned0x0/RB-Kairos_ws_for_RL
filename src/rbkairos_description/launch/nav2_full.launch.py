import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # Param file path
    nav2_params_file = os.path.join(
        get_package_share_directory('rbkairos_description'),
        'config',
        'nav2_params.yaml'
    )

    # Common remappings for all Nav2 nodes
    # Nav2 publishes Twist to /robot/nav2_cmd_vel
    # The twist_relay node converts it to TwistStamped for the motor controller
    remappings = [
        ('cmd_vel', '/robot/nav2_cmd_vel'),
        ('/robot/cmd_vel', '/robot/nav2_cmd_vel'),
        ('odom', '/robot/odom'),
        ('goal_pose', '/goal_pose'),
        ('/robot/goal_pose', '/goal_pose')
    ]

    # --- Twist Relay (converts Twist → TwistStamped for motor controller) ---
    twist_relay_node = Node(
        package='rbkairos_description',
        executable='twist_relay.py',
        name='twist_relay',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    # --- Lifecycle Manager (orchestrates startup order) ---
    lifecycle_manager_node = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_navigation',
        output='screen',
        namespace='robot',
        parameters=[
            {'use_sim_time': True},
            {'autostart': True},
            {'node_names': [
                'controller_server',
                'planner_server',
                'behavior_server',
                'bt_navigator',
                'waypoint_follower'
            ]}
        ]
    )

    # --- Controller Server (local trajectory following - RPP) ---
    controller_server_node = Node(
        package='nav2_controller',
        executable='controller_server',
        name='controller_server',
        output='screen',
        namespace='robot',
        parameters=[nav2_params_file],
        remappings=remappings
    )

    # --- Planner Server (global path planning - NavFn) ---
    planner_server_node = Node(
        package='nav2_planner',
        executable='planner_server',
        name='planner_server',
        output='screen',
        namespace='robot',
        parameters=[nav2_params_file],
        remappings=remappings
    )

    # --- Behavior Server (recovery behaviors: spin, backup, wait) ---
    behavior_server_node = Node(
        package='nav2_behaviors',
        executable='behavior_server',
        name='behavior_server',
        output='screen',
        namespace='robot',
        parameters=[nav2_params_file],
        remappings=remappings
    )

    # --- BT Navigator (Behavior Tree: listens to goal_pose from RViz) ---
    bt_navigator_node = Node(
        package='nav2_bt_navigator',
        executable='bt_navigator',
        name='bt_navigator',
        output='screen',
        namespace='robot',
        parameters=[nav2_params_file],
        remappings=remappings
    )

    # --- Waypoint Follower ---
    waypoint_follower_node = Node(
        package='nav2_waypoint_follower',
        executable='waypoint_follower',
        name='waypoint_follower',
        output='screen',
        namespace='robot',
        parameters=[nav2_params_file],
        remappings=remappings
    )

    return LaunchDescription([
        twist_relay_node,
        controller_server_node,
        planner_server_node,
        behavior_server_node,
        bt_navigator_node,
        waypoint_follower_node,
        lifecycle_manager_node
    ])
