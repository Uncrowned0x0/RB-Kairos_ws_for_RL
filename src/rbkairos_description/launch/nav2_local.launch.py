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

    # Lifecycle Manager node (required for Nav2 nodes)
    lifecycle_manager_node = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_navigation',
        output='screen',
        namespace='robot',
        parameters=[
            {'use_sim_time': True},
            {'autostart': True},
            {'node_names': ['controller_server']}
        ]
    )

    # Controller Server node (includes internal local_costmap)
    controller_server_node = Node(
        package='nav2_controller',
        executable='controller_server',
        name='controller_server',
        output='screen',
        namespace='robot',
        parameters=[nav2_params_file],
        remappings=[
            ('cmd_vel', '/robot/robotnik_base_control/reference_unstamped'),
            ('odom', '/robot/odom')
        ]
    )

    return LaunchDescription([
        lifecycle_manager_node,
        controller_server_node
    ])
