import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Package directories
    robotnik_gazebo_ignition_dir = FindPackageShare('robotnik_gazebo_ignition')
    rbkairos_description_dir = FindPackageShare('rbkairos_description')
    kairos_rl_dir = FindPackageShare('kairos_rl')
    
    # Launch arguments
    gripper_type = LaunchConfiguration('gripper_type', default='schunk_egk50')
    
    # 1. Spawn world
    spawn_world = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([robotnik_gazebo_ignition_dir, 'launch', 'spawn_world.launch.py'])
        ),
        launch_arguments={'world': 'labo'}.items()
    )
    
    # 2. Spawn robot
    spawn_robot = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([robotnik_gazebo_ignition_dir, 'launch', 'spawn_robot.launch.py'])
        ),
        launch_arguments={
            'gripper_type': gripper_type,
            'robot_xacro_path': PathJoinSubstitution([rbkairos_description_dir, 'robots', 'rbkairos_ur5_rl.urdf.xacro']),
            'use_sim': 'true',
            'gazebo_ignition': 'true',
            'robot': 'rbkairos',
            'robot_model': 'rbkairos_plus'
        }.items()
    )
    
    # 3. Train PPO Node
    # 10 second delay to allow Gazebo and controllers to start
    train_ppo_node = TimerAction(
        period=10.0,
        actions=[
            Node(
                package='kairos_rl',
                executable='train_ppo',
                name='train_ppo',
                output='screen',
                # Force execution in same terminal to see training logs
                # The xterm argument can be used with "prefix='xterm -e'" for a separate window
                arguments=[
                    '--config', PathJoinSubstitution([kairos_rl_dir, 'config', 'rl_params.yaml'])
                ]
            )
        ]
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'gripper_type', 
            default_value='schunk_egk50',
            description='Gripper type (schunk_egk50 or tesollo_dg5f)'
        ),
        spawn_world,
        spawn_robot,
        train_ppo_node
    ])
